from abc import ABC, abstractmethod
import gradio as gr
from typing import Any, Dict, Optional, Union, Tuple, AsyncGenerator

class Extension(ABC):
    @abstractmethod
    def init(self, collective_brain) -> None:
        """Initialize the extension with the collective brain instance"""
        pass

    @abstractmethod
    def get_ui_components(self) -> Dict[str, Any]:
        """Return UI components to be added to the interface"""
        pass

    @abstractmethod
    async def pre_process(self, message: str, history: list, **kwargs) -> AsyncGenerator[Union[Tuple[str, list], Tuple[str, list, bool]], None]:
        """Pre-process the message before main processing
        Yields:
            tuple: (message, history) or (message, history, skip_main_inference)
            where skip_main_inference is an optional boolean indicating whether to skip the normal inference
        """
        yield message, history

    @abstractmethod
    def post_process(self, response: str, history: list, **kwargs) -> tuple[str, list]:
        """Post-process the response after main processing"""
        return response, history

    @abstractmethod
    def get_extension_name(self) -> str:
        """Return the name of the extension"""
        pass
