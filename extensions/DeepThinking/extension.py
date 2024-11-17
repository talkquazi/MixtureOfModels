from extensions import Extension
import gradio as gr
from typing import Dict, Any, Optional, AsyncGenerator, Tuple
import json
import asyncio

class DeepThinkingExtension(Extension):
    def __init__(self):
        self.enabled = False
        self.rounds = 5  # Default rounds
        self.vote_threshold = 8
        self.collective_brain = None
        self.current_process = None
        self.initialized = False  # Add initialization tracking
        print("Initializing DeepThinking Extension")

    def init(self, collective_brain) -> None:
        self.collective_brain = collective_brain
        self.initialized = True  # Mark as initialized
        print("DeepThinking Extension connected to collective brain")

    def get_ui_components(self) -> Dict[str, Any]:
        print("Creating DeepThinking UI components")
        
        # Create components without rendering them
        enable = gr.Checkbox(
            label="Deep Thinking",
            value=self.enabled,  # Use instance value
            scale=1,
            min_width=100,
            render=False  # Prevent auto-rendering
        )
        rounds = gr.Slider(
            minimum=2,
            maximum=20,
            value=self.rounds,  # Use instance value
            step=1,
            label="Thinking Rounds",
            scale=2,
            min_width=200,
            render=False  # Prevent auto-rendering
        )
        
        def update_settings(enabled, num_rounds):
            if not self.initialized:
                return
            self.enabled = enabled
            self.rounds = int(num_rounds)  # Ensure integer value
            print(f"DeepThinking settings updated - Enabled: {enabled}, Rounds: {self.rounds}")
        
        # Return components and their layout information
        print("DeepThinking UI components created successfully")
        return {
            "components": [enable, rounds],
            "layout": "row",  # Tell the main app how to arrange them
            "update_fn": update_settings  # Pass the update function
        }

    async def generate_vote(self, character: Dict, message: str, thoughts: str) -> int:
        """Generate a vote from a character on the current round's progress"""
        prompt = f"""
        As {character['name']}, evaluate how well the thinking process is addressing:
        Original Question: {message}
        Current Thoughts: {thoughts}

        Rate on a scale of 1-10 where:
        1 = completely off track or needs significant more thinking
        10 = perfectly addressed and ready for conclusion

        Respond with ONLY your vote number (1-10).
        """
        
        response = await self.collective_brain.process_thought(
            character,
            prompt,
            "",
            None
        )
        
        try:
            # Extract number from response and ensure it's between 1-10
            vote = int(''.join(filter(str.isdigit, response.split()[0])))
            return max(1, min(10, vote))
        except (ValueError, IndexError):
            return 5  # Default middle vote if parsing fails

    async def process_deep_thinking(self, message: str, history: list) -> AsyncGenerator[Tuple[str, list], None]:
        if not self.enabled or not self.initialized:
            yield message, history
            return

        print(f"Starting Deep Thinking process with {self.rounds} rounds")
        last_conclusion = None
        final_history = history.copy()
        
        try:
            for round_num in range(self.rounds):
                print(f"DeepThinking Round {round_num + 1} of {self.rounds}")
                
                # Update chat with current round status
                final_history[-1] = (message, f"DeepThinking Round {round_num + 1} of {self.rounds}")
                yield message, final_history
                
                # Generate responses from each character
                thoughts = []
                for character in self.collective_brain.cognitive_functions:
                    try:
                        # Update chat with current character
                        final_history[-1] = (message, f"Processing {character['thought_process']}...")
                        yield message, final_history
                        
                        response = await self.collective_brain.process_thought(
                            character,
                            message if round_num == 0 else last_conclusion,
                            "\n".join(thoughts),
                            None
                        )
                        thoughts.append(f"{character['name']}: {response}")
                        if self.collective_brain.debug_mode:
                            print(f"{character['name']} thought: {response}")
                    except Exception as e:
                        print(f"Error processing thought for {character['name']}: {str(e)}")
                        continue
                
                if not thoughts:
                    continue
                
                # Collect votes
                votes = []
                for character in self.collective_brain.cognitive_functions:
                    try:
                        vote = await self.generate_vote(character, message, "\n".join(thoughts))
                        votes.append(vote)
                        if self.collective_brain.debug_mode:
                            print(f"{character['name']} votes: {vote}/10")
                    except Exception as e:
                        print(f"Error collecting vote from {character['name']}: {str(e)}")
                        continue
                
                if not votes:
                    continue
                
                avg_vote = sum(votes) / len(votes)
                print(f"Round {round_num + 1} average vote: {avg_vote:.1f}/10")
                
                # Synthesize thoughts
                try:
                    round_conclusion = await self.collective_brain.synthesize_thoughts(
                        message,
                        "\n".join(thoughts)
                    )
                    last_conclusion = round_conclusion
                    
                    # Update chat with interim conclusion
                    final_history[-1] = (message, f"Round {round_num + 1} conclusion: {round_conclusion}")
                    yield message, final_history
                    
                    if avg_vote >= self.vote_threshold:
                        print(f"Reached satisfactory conclusion with average vote of {avg_vote:.1f}/10")
                        final_history[-1] = (message, round_conclusion)
                        yield message, final_history
                        return
                        
                    if round_num == self.rounds - 1:
                        print(f"Final round reached with average vote of {avg_vote:.1f}/10")
                        final_history[-1] = (message, round_conclusion)
                        yield message, final_history
                        return
                except Exception as e:
                    print(f"Error synthesizing thoughts: {str(e)}")
                    continue
                
                await asyncio.sleep(0.1)  # Prevent event loop starvation
            
            # If we get here, yield the last conclusion
            if last_conclusion:
                final_history[-1] = (message, last_conclusion)
            yield message, final_history
            
        except Exception as e:
            print(f"Error in deep thinking process: {str(e)}")
            yield message, final_history

    async def pre_process(self, message: str, history: list, **kwargs) -> AsyncGenerator[Tuple[str, list, bool], None]:
        try:
            if self.enabled:
                history = history or []
                if not history or message != history[-1][0]:
                    history.append((message, ""))
                
                async for new_message, new_history in self.process_deep_thinking(message, history):
                    yield new_message, new_history, True
            else:
                yield message, history, False
        except Exception as e:
            print(f"Error in DeepThinking pre-process: {str(e)}")
            yield message, history, False

    async def post_process(self, response: str, history: list, **kwargs) -> tuple[str, list]:
        return response, history

    def get_extension_name(self) -> str:
        return "DeepThinking"
