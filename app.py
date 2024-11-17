import json
import openai
from typing import List, Dict
import time
import shutil
import sys
import threading
from datetime import datetime
import gradio as gr
import base64
from io import BytesIO
from PIL import Image
import io  # Add this import at the top
import asyncio  # Add this import at the top
import subprocess  # Add this import for nvidia-smi
import logging  # Add this for temperature logging
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv
from extensions import Extension  # Add this import

# Load environment variables
load_dotenv()

# Environment Configuration
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/v1/')
GRADIO_PORT = int(os.getenv('GRADIO_PORT', 7860))
GPU_TEMP_LIMIT = int(os.getenv('GPU_TEMP_LIMIT', 75))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
TOKEN_MODE = os.getenv('TOKEN_MODE', 'True').lower() == 'true'

# Collective AI Chat Interface
# This script demonstrates a chat interface for a collective AI brain that combines multiple models
# to generate a unified response. The collective brain consists of multiple cognitive functions that
# each provide a unique perspective on the input query. The central synthesizer then combines these
# perspectives into a cohesive response. The script uses OpenAI's Chat API to interact with the models.
# AKA: MOM - Mixture Of Models

DEBUG_MODE = False
TOKEN_MODE = True

class BorderFormatter:
    @staticmethod
    def get_terminal_width():
        return shutil.get_terminal_size().columns - 4

    @staticmethod
    def format_output(title: str, message: str, style: str = "normal"):
        width = BorderFormatter.get_terminal_width()
        box_chars = {
            "normal": {"tl": "╭", "tr": "╮", "bl": "╯", "br": "╰", "h": "─", "v": "│"},
            "thick": {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║"}
        }[style]
        
        # Split and wrap message
        lines = []
        for line in message.split('\n'):
            while line and len(line) > width - 4:
                split_at = line[:width-4].rfind(' ')
                if split_at == -1:
                    split_at = width-4
                lines.append(line[:split_at])
                line = line[split_at:].lstrip()
            if line:
                lines.append(line)
        
        # Format with border
        border_top = f"{box_chars['tl']}{box_chars['h'] * (width - 2)}{box_chars['tr']}"
        border_bottom = f"{box_chars['bl']}{box_chars['h'] * (width - 2)}{box_chars['br']}"
        
        formatted = [border_top]
        formatted.append(f"{box_chars['v']} {title:<{width-4}} {box_chars['v']}")
        formatted.append(f"{box_chars['v']}{box_chars['h'] * (width - 2)}{box_chars['v']}")
        
        for line in lines:
            formatted.append(f"{box_chars['v']} {line:<{width-4}} {box_chars['v']}")
            
        formatted.append(border_bottom)
        return "\n".join(formatted)

class ThinkingAnimation:
    def __init__(self, text):
        self.text = text
        self.running = False
        self.thread = None

    def animate(self):
        dots = 1
        while self.running:
            sys.stdout.write(f'\r{self.text} {"." * dots:<3}')
            sys.stdout.flush()
            dots = (dots % 3) + 1
            time.sleep(0.5)
        sys.stdout.write('\r' + ' ' * (len(self.text) + 4) + '\r')
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.animate)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

class TokenTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.token_count = 0
        self.generation_segments = []

    def start(self):
        self.start_time = time.time()
        self.token_count = 0
        self.generation_segments = []

    def add_tokens(self, text: str, segment_name: str):
        """Count tokens (words) in text and add to segment tracking"""
        tokens = len(text.split())
        self.token_count += tokens
        self.generation_segments.append({
            'name': segment_name,
            'tokens': tokens,
            'time': time.time() - self.start_time
        })

    def stop(self):
        self.end_time = time.time()

    def get_metrics(self):
        total_time = self.end_time - self.start_time
        tokens_per_second = self.token_count / total_time if total_time > 0 else 0
        
        metrics = f"\n=== Generation Metrics ===\n"
        metrics += f"Total time: {total_time:.2f}s\n"
        metrics += f"Total tokens: {self.token_count}\n"
        metrics += f"Tokens per second: {tokens_per_second:.2f}\n\n"
        metrics += "Generation segments:\n"
        
        for segment in self.generation_segments:
            metrics += f"- {segment['name']}: {segment['tokens']} tokens at {segment['time']:.2f}s\n"
        
        return metrics

class GPUMonitor:
    def __init__(self, temp_limit=GPU_TEMP_LIMIT):
        self.temp_limit = temp_limit
        self.logger = logging.getLogger('GPUMonitor')
        self.logger.setLevel(logging.INFO)
        
    def get_gpu_temperature(self) -> float:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            return float(result.strip())
        except Exception as e:
            self.logger.warning(f"Failed to get GPU temperature: {e}")
            return 0.0
            
    async def check_and_cool(self, pre_or_post: str = "pre") -> None:
        temp = self.get_gpu_temperature()
        self.logger.info(f"{pre_or_post}-inference GPU Temperature: {temp}°C")
        
        if temp > self.temp_limit:
            self.logger.warning(f"GPU temperature {temp}°C exceeded limit of {self.temp_limit}°C. Cooling down...")
            await asyncio.sleep(30)
            new_temp = self.get_gpu_temperature()
            self.logger.info(f"After cooling, GPU Temperature: {new_temp}°C")

class CollectiveBrain:
    def __init__(self, base_url: str = OLLAMA_API_URL, api_key: str = '1234'):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self._load_cognitive_functions()
        self.system_prompt = "You are part of a collective AI brain working together as one mind."
        self.internal_history = []  # Stores collective thought process history
        self.human_history = []     # Stores human-like response history
        self.simple_greetings = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'sup', 'what\'s up', 'yo', 'hey there'}
        self.tracker = TokenTracker()
        self.chat_history = []
        self.debug_mode = DEBUG_MODE  # Instance-level debug mode
        self.gpu_monitor = GPUMonitor()  # Add GPU monitor to existing initialization
        self.extensions = {}
        self.load_extensions()

    def set_debug_mode(self, enabled: bool):
        """Update debug mode at runtime"""
        self.debug_mode = enabled
        print(f"Debug mode {'enabled' if enabled else 'disabled'}")

    def _load_cognitive_functions(self):
        with open('characters.json', 'r') as f:
            data = json.load(f)
            self.cognitive_functions = data['characters']
            self.synthesizer = data['conclusion_thinker']
            self.human_output = data['human_output']

    def _sort_cognitive_functions(self, image_path: str = None):
        """Sort cognitive functions to prioritize vision models when image is present"""
        if not image_path:
            return self.cognitive_functions
            
        vision_first = []
        others = []
        
        for func in self.cognitive_functions:
            if any('vision' in interest.lower() for interest in func['interests']):
                vision_first.append(func)
            else:
                others.append(func)
                
        return vision_first + others

    def is_simple_input(self, input_data: str) -> bool:
        """Determine if the input is a simple greeting or basic query"""
        input_lower = input_data.lower().strip()
        return (
            input_lower in self.simple_greetings or 
            len(input_data.split()) <= 3
        )

    def humanize_response(self, input_data: str, collective_response: str) -> str:
        if self.is_simple_input(input_data):
            prompt = f"""
            Transform this response into a brief, friendly greeting:
            {collective_response}
            Keep it natural and concise, no need for elaboration unless asked to do so.
            """
        else:
            history_context = "\n".join([f"Human: {h['input']}\nAssistant: {h['response']}" 
                                   for h in self.human_history[-5:]])

            prompt = f"""
            Previous Human Interactions:
            {history_context}

            User Input: {input_data}
            Internal Collective Response: {collective_response}

            Transform the collective response into a more natural, human-like response while maintaining
            the core information and helpfulness. Make it conversational and engaging while ensuring
            all key points are preserved to answer the initial input from the Human input and Human input history.
            Be sure to stay focused on the Human: input. Do not reference other AI names in the output.
            """

        animation = ThinkingAnimation("Humanizing Response")
        animation.start()
        try:
            response = self.client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=self.human_output['model']
            )
            result = response.choices[0].message.content
            self.tracker.add_tokens(result, "Human Output")
            return result
        finally:
            animation.stop()

    async def process_input_with_progress(self, message: str, history: List[List[str]], image_path: str = None) -> tuple[str, str]:
        self.tracker.start()
        
        # Convert Gradio history format to internal format
        self.human_history = [
            {'input': h[0], 'response': h[1]} for h in history
        ]
        
        if self.debug_mode:
            print(BorderFormatter.format_output("Input Query", message, "thick"))
            if (image_path):
                print(BorderFormatter.format_output("Image Input", "Processing with vision models first", "thick"))
        
        # Sort cognitive functions to prioritize vision models if image is present
        sorted_functions = self._sort_cognitive_functions(image_path)
        collected_thoughts = []
        vision_description = None
        
        # Send initial progress update
        yield "Processing input..."
        
        for function in sorted_functions:
            yield f"{function['thought_process']}..."
            # For non-vision models, include the vision description if available
            if vision_description and not any('vision' in interest.lower() for interest in function['interests']):
                context = f"Vision Analysis: {vision_description}\n\nBased on this visual context, {message}"
            else:
                context = message
                
            thought = await self.process_thought(function, context, "\n".join(collected_thoughts), image_path)
            collected_thoughts.append(f"{function['name']} Processing: {thought}")
            
            # Store the first vision model's description
            if image_path and vision_description is None and any('vision' in interest.lower() for interest in function['interests']):
                vision_description = thought
            
            if self.debug_mode:
                print(BorderFormatter.format_output(function['name'], thought))
                print()
            await asyncio.sleep(0.1)  # Simulate async progress

        yield "Synthesizing collective response..."
        
        # Synthesize collective response
        if self.debug_mode:
            print("\n=== Synthesizing Collective Response ===\n")
        
        collective_response = await self.synthesize_thoughts(message, "\n".join(collected_thoughts))
        # Store in internal history
        self.internal_history.append({
            'input': message,
            'response': collective_response,
            'thoughts': collected_thoughts
        })
        
        if self.debug_mode:
            print("\n" + BorderFormatter.format_output("Collective Response", collective_response, "normal"))
        
        # Transform to human-like response
        human_response = await self.humanize_response(message, collective_response)
        # Store in human interaction history
        self.human_history.append({
            'input': message,
            'response': human_response
        })

        self.tracker.stop()
        metrics = self.tracker.get_metrics() if TOKEN_MODE else ""
        yield human_response, metrics

    async def process_thought(self, cognitive_function: Dict, input_data: str, previous_thoughts: str, image_path: str = None) -> str:
        animation = ThinkingAnimation(f"{cognitive_function['thought_process']}")
        animation.start()
        
        is_vision_function = any('vision' in interest.lower() for interest in cognitive_function['interests'])
        
        try:
            # Check temperature before inference
            await self.gpu_monitor.check_and_cool("pre")
            
            # Start the token tracker if not already started
            if self.tracker.start_time is None:
                self.tracker.start()
            
            if image_path and is_vision_function:
                # Format message specifically for vision models
                img_base64 = self._encode_image_to_base64(image_path)
                if img_base64:
                    messages = [{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text', 
                                'text': f"""
                                Analyze the following image and provide a highly detailed, natural response.
                                From the user's query of: {input_data}.
                                """,
                                'type': 'image_url', 
                                'image_url': f"data:image/jpeg;base64,{img_base64}"
                            }
                        ]
                    }]
                    
                    if self.debug_mode:
                        print(f"Vision message format: {messages}")
            else:
                # Internal thoughts about an image input
                if image_path:
                    prompt = f"""
                    As {cognitive_function['name']}, provide a brief, natural response about the image described.
                    Keep it simple and direct, focusing on {cognitive_function['personality']} while maintaining context
                    with the user's original input of: {input_data}.
                    """
                # Regular non-vision message handling
                elif self.is_simple_input(input_data):
                    prompt = f"""
                    As {cognitive_function['name']}, provide a brief, natural response to: {input_data}
                    Keep it simple and direct, focusing on {cognitive_function['personality']}.
                    """
                else:
                    history_context = "\n".join([f"Input: {h['input']}\nThoughts: {h['thoughts'][-1] if h['thoughts'] else ''}" 
                                           for h in self.internal_history[-3:]])
                    
                    prompt = f"""
                    As {cognitive_function['name']}, analyze the following input and context:

                    Previous Context:
                    {history_context}

                    Current Input: {input_data}
                    Previous Thoughts in Current Process: {previous_thoughts}

                    Focus on your specific cognitive aspect: {cognitive_function['interests']}
                    Provide your specialized analysis while maintaining awareness of the collective mind.
                    Be sure to stay on topic with the Current Input if you think your cognitive aspect provides no insight say so.
                    But you are not limited to your specific cognitive aspect if the topic is out of your interests.
                    Give your response for the Current Input.
                    """
                
                messages = [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ]

            # Make API request
            response = self.client.chat.completions.create(
                model=cognitive_function['model'],
                messages=messages
            )
            
            # Check temperature after inference
            await self.gpu_monitor.check_and_cool("post")
            
            thought = response.choices[0].message.content
            self.tracker.add_tokens(thought, f"{cognitive_function['name']} thought")
            return thought
            
        finally:
            animation.stop()

    async def synthesize_thoughts(self, input_data: str, collected_thoughts: str) -> str:
        # Check temperature before synthesis
        await self.gpu_monitor.check_and_cool("pre")
        
        if self.is_simple_input(input_data):
            prompt = f"""
            Provide a brief, natural response to: {input_data}
            Consider the collected thoughts but keep the response simple and friendly.
            Collected thoughts: {collected_thoughts}
            """
        else:
            history_context = "\n".join([
                f"Input: {h['input']}\nCollective Response: {h['response']}" 
                for h in self.internal_history[-3:]
            ])
        
            prompt = f"""
            As the central synthesizer, integrate all cognitive processing into a collective response.
            
            Previous Internal Processing:
            {history_context}

            Current Input: {input_data}
            Current Cognitive Processing:
            {collected_thoughts}
            
            Synthesize a unified response that represents our collective intelligence.
            Focus on addressing the input query while maintaining continuity with previous thoughts.
            """
        
        response = self.client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            model=self.synthesizer['model']
        )
        
        # Check temperature after synthesis
        await self.gpu_monitor.check_and_cool("post")
        
        result = response.choices[0].message.content
        self.tracker.add_tokens(result, "Synthesis")
        return result

    async def humanize_response(self, input_data: str, collective_response: str) -> str:
        # Check temperature before humanization
        await self.gpu_monitor.check_and_cool("pre")
        
        if self.is_simple_input(input_data):
            prompt = f"""
            Based on the Human input transform the collective response into a brief, friendly greeting:

            ```human_input
            {input_data}
            ```

            ```collective_response
            {collective_response}
            
            Keep it natural and concise, no need for elaboration unless asked to do so.```
            """
        else:
            history_context = "\n".join([f"Human: {h['input']}\nAssistant: {h['response']}" 
                                   for h in self.human_history[-5:]])

            prompt = f"""
            Previous Human Interactions:
            {history_context}

            User Input: {input_data}
            Internal Collective Response: {collective_response}

            Transform the collective response into a more natural, human-like response while maintaining
            the core information and helpfulness. Make it conversational and engaging while ensuring
            all key points are preserved.
            """

        animation = ThinkingAnimation("Humanizing Response")
        animation.start()
        try:
            response = self.client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=self.human_output['model']
            )
            
            # Check temperature after humanization
            await self.gpu_monitor.check_and_cool("post")
            
            result = response.choices[0].message.content
            self.tracker.add_tokens(result, "Human Output")
            return result
        finally:
            animation.stop()

    def clear_history(self):
        """Clear both internal and human interaction histories"""
        self.internal_history = []
        self.human_history = []
        print("All conversation histories cleared.")

    def _encode_image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with Image.open(image_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            if self.debug_mode:
                print(f"Error encoding image: {str(e)}")
            return None

    def load_extensions(self):
        """Load all extensions from the extensions directory"""
        print("\n=== Loading Extensions ===")
        extensions_dir = Path(__file__).parent / 'extensions'
        if not extensions_dir.exists():
            print("No extensions directory found")
            return

        for ext_dir in extensions_dir.iterdir():
            if ext_dir.is_dir() and not ext_dir.name.startswith('__'):
                try:
                    print(f"Loading extension from directory: {ext_dir.name}")
                    # Import the extension module
                    module = importlib.import_module(f'extensions.{ext_dir.name}.extension')
                    
                    # Find the Extension class
                    ext_class = None
                    for item in module.__dict__.values():
                        if isinstance(item, type) and issubclass(item, Extension) and item != Extension:
                            ext_class = item
                            break
                    
                    if ext_class:
                        extension = ext_class()
                        extension.init(self)
                        self.extensions[extension.get_extension_name()] = extension
                        print(f"✓ Successfully loaded extension: {extension.get_extension_name()}")
                    else:
                        print(f"✗ No valid extension class found in {ext_dir.name}")
                except Exception as e:
                    print(f"✗ Failed to load extension {ext_dir.name}: {str(e)}")
        print("=== Extension Loading Complete ===\n")

def create_chat_interface(collective_brain: CollectiveBrain):
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css="""
        .contains-code {
            font-family: monospace;
            background-color: #f6f6f6;
            padding: 10px;
            border-radius: 4px;
        }
        """
    ) as demo:
        gr.Markdown("""
        # Mixture Of Models
        Welcome to the Collective AI brain - a unified intelligence combining multiple cognitive functions via mixture of models.
        """)
        
        with gr.Row():
            debug_toggle = gr.Checkbox(
                label="Debug Mode",
                value=DEBUG_MODE,
                interactive=True,
            )
            
        chatbot = gr.Chatbot(
            show_label=False,
            height=500,  # Reduced height to make room for stats
            container=True,
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                show_label=False,
                scale=8,
            )
            image_input = gr.Image(
                type="filepath",
                label="Attach Image",
                scale=1,
            )
            submit = gr.Button(
                "Send",
                scale=1,
                variant="primary",
            )
        
        # Add extension UI components in a single place
        for extension in collective_brain.extensions.values():
            try:
                print(f"Adding UI components for extension: {extension.get_extension_name()}")
                components = extension.get_ui_components()
                if isinstance(components, dict) and "components" in components:
                    # Create a container based on layout
                    if components["layout"] == "row":
                        with gr.Row():
                            rendered_components = [comp.render() for comp in components["components"]]
                    else:  # Default to group
                        with gr.Group():
                            rendered_components = [comp.render() for comp in components["components"]]
                            
                    # Set up any event handlers
                    if "update_fn" in components and len(rendered_components) >= 2:
                        rendered_components[0].change(
                            components["update_fn"],
                            inputs=rendered_components,
                            outputs=[]
                        )
                    
                    print(f"✓ Successfully added UI components for {extension.get_extension_name()}")
            except Exception as e:
                print(f"✗ Failed to add UI components for {extension.get_extension_name()}: {str(e)}")
            
        # Add stats display area
        stats_display = gr.Markdown(visible=TOKEN_MODE)
            
        with gr.Row():
            clear = gr.Button("Clear Chat")

        async def respond(message, history, image):
            if message.strip() == "":
                yield "", history, ""
                return
                
            history = history or []
            history.append((message, ""))
            
            # Pre-process through extensions
            skip_main_inference = False
            for extension in collective_brain.extensions.values():
                try:
                    # Handle async generator from pre_process
                    async for result in extension.pre_process(message, history):
                        if len(result) == 3:  # Extension returned skip_main_inference flag
                            new_message, new_history, should_skip = result
                            message = new_message
                            history = new_history
                            skip_main_inference = skip_main_inference or should_skip
                            # Update chat with intermediate results
                            yield "", history, ""
                        else:
                            message, history = result
                            yield "", history, ""
                except Exception as e:
                    print(f"Extension {extension.get_extension_name()} pre-process error: {e}")

            if not skip_main_inference:
                response_generator = collective_brain.process_input_with_progress(message, history, image)
                response_text = ""
                metrics = ""
                
                async for progress in response_generator:
                    if isinstance(progress, tuple):
                        response_text, metrics = progress
                    else:
                        history[-1] = (message, progress)
                        yield "", history, ""
                        await asyncio.sleep(0.1)

                # Post-process through extensions
                for extension in collective_brain.extensions.values():
                    try:
                        response_text, history = await extension.post_process(response_text, history)
                    except Exception as e:
                        print(f"Extension {extension.get_extension_name()} post-process error: {e}")

                history[-1] = (message, response_text)
                yield "", history, metrics
            else:
                # If an extension handled the inference, just return the current state
                yield "", history, ""

        msg.submit(
            respond,
            [msg, chatbot, image_input],
            [msg, chatbot, stats_display],
            api_name=False
        )
        submit.click(
            respond,
            [msg, chatbot, image_input],
            [msg, chatbot, stats_display],
            api_name=False
        )
            
        def clear_chat():
            collective_brain.clear_history()
            return [], ""  # Clear chat history and stats display
            
        def toggle_debug(enabled):
            collective_brain.set_debug_mode(enabled)
            
        debug_toggle.change(
            toggle_debug,
            inputs=[debug_toggle],
            outputs=[],
        )
        
        clear.click(clear_chat, None, [chatbot, stats_display])
        
    return demo

if __name__ == "__main__":
    collective_brain = CollectiveBrain()
    demo = create_chat_interface(collective_brain)
    demo.queue()  # Enable queuing
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_api=False
    )
