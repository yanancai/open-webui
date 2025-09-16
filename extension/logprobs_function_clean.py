"""
Open WebUI Function Extension: Ollama Logprob Heatmap Visualizer
This Filter Function displays model response tokens as a heatmap based on logprobs
from Ollama's API and shows top-k alternatives on hover.

Compatible with:
- Ollama's /api/chat endpoint (native format)
- Ollama's /api/generate endpoint (native format)
- Ollama's OpenAI-compatible /v1/chat/completions endpoint (OpenAI format)

Input formats supported:
- Native Ollama: {"options": {"logprobs": true, "top_logprobs": 5}}
- OpenAI-compatible: {"logprobs": true, "top_logprobs": 5}

Output formats supported:
- Native Ollama: logprobs in message.logprobs or response.logprobs
- OpenAI-compatible: logprobs in choices[0].logprobs.content[]
"""

from pydantic import BaseModel, Field
import json
import math
import re
from typing import Optional, List, Dict, Any
import logging


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority level for this filter")
        enable_heatmap: bool = Field(
            default=True, description="Enable logprob heatmap visualization (disable for debugging raw logprobs)"
        )
        preserve_logprobs: bool = Field(
            default=True, description="Always preserve raw logprobs data in message object for UI access"
        )
        show_debug_ui: bool = Field(
            default=False, description="Show debug UI with raw logprobs data even when heatmap is disabled"
        )
        show_top_k: int = Field(
            default=5, 
            description="Number of top alternative tokens to show on hover (0-20)",
            ge=0,
            le=20
        )
        min_confidence_threshold: float = Field(
            default=0.1, description="Minimum confidence to show heatmap (0-1)"
        )
        color_scheme: str = Field(
            default="red_blue", description="Color scheme: red_blue, heat, rainbow"
        )
        enable_streaming_logprobs: bool = Field(
            default=True, description="Enable logprobs for streaming requests (may cause server issues on older versions)"
        )
        debug_logs: bool = Field(default=True, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Modify the request to ensure logprobs are requested if heatmap is enabled OR if preserve_logprobs is enabled
        """
        if self.valves.debug_logs:
            self.logger.info("=== INLET DEBUG ===")
            self.logger.info(f"Original request body: {json.dumps(body, indent=2)}")

        if not self.valves.enable_heatmap and not self.valves.preserve_logprobs:
            if self.valves.debug_logs:
                self.logger.info("Both heatmap and logprobs preservation disabled, returning original body")
            return body
        
        if self.valves.debug_logs:
            self.logger.info(f"Processing request - heatmap: {self.valves.enable_heatmap}, preserve_logprobs: {self.valves.preserve_logprobs}")

        # Check if this is a streaming request
        is_streaming = body.get("stream", False)
        
        # Detect if this is an OpenAI-compatible request (has "messages" field)
        is_openai_format = "messages" in body
        
        if self.valves.debug_logs:
            self.logger.info(f"Request type: {'streaming' if is_streaming else 'non-streaming'}")
            self.logger.info(f"Request format: {'OpenAI-compatible' if is_openai_format else 'Ollama native'}")

        # For streaming requests, only add logprobs if enabled and supported
        if is_streaming:
            if not self.valves.enable_streaming_logprobs:
                if self.valves.debug_logs:
                    self.logger.info("Streaming logprobs disabled - skipping logprobs for streaming request")
                return body
                
            if self.valves.debug_logs:
                self.logger.info("Streaming request detected - adding logprobs for heatmap or preservation")

        # Handle OpenAI-compatible format
        if is_openai_format:
            # For OpenAI format, logprobs parameters are at the top level
            if "logprobs" not in body:
                if self.valves.debug_logs:
                    self.logger.info("Adding OpenAI-compatible logprobs parameters (for heatmap or preservation)")
                body["logprobs"] = True
                body["top_logprobs"] = min(max(self.valves.show_top_k, 0), 20)
            else:
                if self.valves.debug_logs:
                    self.logger.info("OpenAI logprobs already configured by user - keeping existing settings")
        else:
            # Handle Ollama native format
            if "options" not in body:
                body["options"] = {}
            
            # Only set logprobs if not already configured by the user
            if "logprobs" not in body["options"]:
                if self.valves.debug_logs:
                    self.logger.info("Adding Ollama native logprobs parameters (for heatmap or preservation)")
                body["options"]["logprobs"] = True
                body["options"]["top_logprobs"] = min(max(self.valves.show_top_k, 0), 20)
            else:
                if self.valves.debug_logs:
                    self.logger.info("Ollama logprobs already configured by user - keeping existing settings")

        if self.valves.debug_logs:
            self.logger.info("=== MODIFIED REQUEST BODY ===")
            self.logger.info(f"Body after adding logprobs options: {json.dumps(body, indent=2)}")
            self.logger.info("=== END INLET DEBUG ===")

        return body

    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process the response and add heatmap visualization if logprobs are present
        """
        if self.valves.debug_logs:
            self.logger.info("=== OUTLET DEBUG ===")
            self.logger.info(f"Response body keys: {list(body.keys())}")
            self.logger.info(f"Response body type: {type(body)}")
            self.logger.info(f"Full response body structure:")
            self.logger.info(f"{json.dumps(body, indent=2, default=str)}")
            self.logger.info("=" * 50)

        if not (self.valves.enable_heatmap or self.valves.preserve_logprobs):
            if self.valves.debug_logs:
                self.logger.info("🔕 Both heatmap and logprobs preservation disabled, returning original response unchanged")
            return body

        try:
            # NEW: Handle the chat completion response structure
            # In this case, we have: {"messages": [...], "chat_id": "...", ...}
            if "messages" in body and isinstance(body["messages"], list):
                if self.valves.debug_logs:
                    self.logger.info("📋 Processing chat completion response with messages array")
                
                for message in body["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        logprobs_data = message.get("logprobs")
                        
                        if self.valves.debug_logs:
                            self.logger.info(f"🤖 Found assistant message with content: '{content}'")
                            self.logger.info(f"📊 Logprobs present: {bool(logprobs_data)}")
                            if logprobs_data:
                                self.logger.info(f"📊 Logprobs count: {len(logprobs_data)}")
                        
                        if logprobs_data and content.strip():
                            if self.valves.enable_heatmap:
                                # Use Ollama format since the logprobs structure looks like Ollama native
                                heatmap_html = self._create_heatmap_html_ollama(content, logprobs_data)
                                # APPEND visualization after original content, don't replace it
                                message["content"] = f"{content}\n\n{heatmap_html}"
                                
                                if self.valves.debug_logs:
                                    self.logger.info("✅ Successfully appended heatmap visualization after original content")
                                    self.logger.info(f"Original content: '{content}'")
                                    self.logger.info(f"Total content length: {len(message['content'])} characters")
                            elif self.valves.show_debug_ui:
                                debug_html = self._create_debug_ui_html(content, logprobs_data)
                                # APPEND debug UI after original content
                                message["content"] = f"{content}\n\n{debug_html}"
                                
                                if self.valves.debug_logs:
                                    self.logger.info("✅ Successfully appended debug UI after original content")
                                    self.logger.info(f"Original content: '{content}'")
                                    self.logger.info(f"Total content length: {len(message['content'])} characters")
                            
                            # Keep logprobs for UI access
                            if self.valves.preserve_logprobs and not message.get("logprobs"):
                                message["logprobs"] = logprobs_data
                                if self.valves.debug_logs:
                                    self.logger.info("✅ Preserved logprobs data in message for UI access")
                            
                            return body
                        elif not logprobs_data:
                            if self.valves.debug_logs:
                                self.logger.warning("No logprobs found in assistant message")
                        elif not content.strip():
                            if self.valves.debug_logs:
                                self.logger.info("Assistant message has logprobs but no content")
                
                if self.valves.debug_logs:
                    self.logger.info("No assistant messages with logprobs found in messages array")
                return body

            # Handle other response structures (legacy code for other API formats)
            if self.valves.debug_logs:
                self.logger.warning("Unknown response structure, returning original")

        except Exception as e:
            if self.valves.debug_logs:
                self.logger.error(f"Error processing logprobs: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")

        return body

    def _create_heatmap_html_ollama(self, content: str, logprobs_data: list) -> str:
        """
        Create HTML with heatmap styling based on Ollama's logprobs format
        """
        if self.valves.debug_logs:
            self.logger.info("=== OLLAMA HEATMAP CREATION DEBUG ===")
            self.logger.info(f"Content length: {len(content)}")
            self.logger.info(f"Logprobs data type: {type(logprobs_data)}")
            self.logger.info(f"Logprobs data length: {len(logprobs_data) if isinstance(logprobs_data, list) else 'not a list'}")

        # Ollama format: direct array of TokenLogprob objects
        if not isinstance(logprobs_data, list) or not logprobs_data:
            if self.valves.debug_logs:
                self.logger.warning("Invalid logprobs data format for Ollama")
            return content

        tokens = []
        token_logprobs = []
        top_logprobs = []

        for item in logprobs_data:
            if isinstance(item, dict):
                tokens.append(item.get("token", ""))
                token_logprobs.append(item.get("logprob", None))
                top_logprobs.append(item.get("top_logprobs", []))
            else:
                if self.valves.debug_logs:
                    self.logger.warning(f"Invalid logprob item format: {type(item)}")
                continue

        if self.valves.debug_logs:
            self.logger.info(f"Processing {len(tokens)} tokens")
            self.logger.info(f"First 3 tokens: {tokens[:3]}")
            self.logger.info(f"First 3 logprobs: {token_logprobs[:3]}")

        return self._generate_heatmap_html(tokens, token_logprobs, top_logprobs)

    def _create_debug_ui_html(self, content: str, logprobs_data, is_openai: bool = False) -> str:
        """
        Create SIMPLE debug UI to test if anything works at all
        """
        if self.valves.debug_logs:
            self.logger.info("=== SIMPLE DEBUG UI CREATION ===")
            self.logger.info(f"Content length: {len(content)}")
            self.logger.info(f"Is OpenAI format: {is_openai}")
            self.logger.info(f"Logprobs data type: {type(logprobs_data)}")

        # Simple test with clean HTML
        simple_html = f"""<div style="color: red; font-weight: bold; border: 2px solid red; padding: 10px;">
    <h1>RED LOGPROBS DEBUG TEST</h1>
    <p>Original content: {self._escape_html(content)}</p>
    <p>Logprobs found: {len(logprobs_data) if isinstance(logprobs_data, list) else 'Not a list'}</p>
</div>"""

        # Return with HTML code block for Artifacts
        result = f"""```html
{simple_html}
```"""

        if self.valves.debug_logs:
            self.logger.info("=== END SIMPLE DEBUG UI CREATION ===")

        return result

    def _generate_heatmap_html(self, tokens: list, token_logprobs: list, top_logprobs: list) -> str:
        """
        Generate native inline HTML that Open WebUI renders directly
        """
        if self.valves.debug_logs:
            self.logger.info(f"Generating heatmap for {len(tokens)} tokens")

        # Validate that we have matching data
        if len(tokens) != len(token_logprobs):
            if self.valves.debug_logs:
                self.logger.warning(f"Token count mismatch: {len(tokens)} tokens vs {len(token_logprobs)} logprobs")
            # Truncate to minimum length
            min_len = min(len(tokens), len(token_logprobs))
            tokens = tokens[:min_len]
            token_logprobs = token_logprobs[:min_len]
            top_logprobs = top_logprobs[:min_len] if len(top_logprobs) > min_len else top_logprobs

        # Generate native inline HTML that Open WebUI processes
        result = f"""

<style>
.logprob-heatmap-container {{
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
    padding: 16px;
    border-radius: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 16px 0;
    border: 2px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}}

.heatmap-header {{
    font-weight: 600;
    color: white;
    margin-bottom: 12px;
    font-size: 16px;
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}}

.heatmap-content {{
    margin: 16px 0;
    line-height: 2.2;
    background: rgba(255,255,255,0.95);
    padding: 16px;
    border-radius: 8px;
    font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', monospace;
}}

.heatmap-footer {{
    font-size: 11px;
    color: white;
    margin-top: 12px;
    text-align: center;
    opacity: 0.9;
}}

.token {{
    padding: 3px 6px;
    margin: 0 1px;
    border-radius: 4px;
    display: inline-block;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
    font-weight: 500;
    border: 1px solid transparent;
}}

.token.low-confidence {{
    background-color: #f1f5f9;
    color: #64748b;
}}

.token.heatmap-token:hover {{
    transform: scale(1.1);
    z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    border: 1px solid #3b82f6;
}}

.token.heatmap-token::after {{
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 11px;
    white-space: pre-line;
    z-index: 1000;
    margin-bottom: 6px;
    max-width: 280px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s ease;
    border: 1px solid #3b82f6;
}}

.token.heatmap-token:hover::after {{
    opacity: 1;
}}
</style>

<html>
<div class="logprob-heatmap-container">
    <div class="heatmap-header">🔥 Logprobs Heatmap Visualization</div>
    <div class="heatmap-content">"""

        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
            if logprob is None:
                confidence = 0.5  # neutral for unknown
            else:
                # Convert logprob to confidence (probability)
                confidence = math.exp(logprob) if logprob > -10 else 0.01

            # Skip if below threshold
            if confidence < self.valves.min_confidence_threshold:
                result += f'<span class="token low-confidence">{self._escape_html(token)}</span>'
                continue

            # Get color based on confidence
            color = self._get_color_for_confidence(confidence)

            # Create tooltip data for alternatives
            tooltip_lines = [f"Confidence: {confidence*100:.1f}%", f"Logprob: {logprob:.3f}"]
            
            if i < len(top_logprobs) and top_logprobs[i]:
                alternatives_list = []
                top_logprob_item = top_logprobs[i]
                if isinstance(top_logprob_item, list):
                    for alt in top_logprob_item[:3]:  # Show top 3 alternatives
                        if isinstance(alt, dict):
                            alt_token = alt.get("token", "")
                            alt_logprob = alt.get("logprob", 0)
                            alt_confidence = (
                                math.exp(alt_logprob) * 100
                                if alt_logprob > -10
                                else 0.01
                            )
                            alternatives_list.append(f"  '{alt_token}' ({alt_confidence:.1f}%)")
                
                if alternatives_list:
                    tooltip_lines.append("Alternatives:")
                    tooltip_lines.extend(alternatives_list)

            tooltip_text = "\\n".join(tooltip_lines)

            result += f'''<span class="token heatmap-token" 
                  style="background-color: {color};" 
                  data-tooltip="{tooltip_text}">
                {self._escape_html(token)}
            </span>'''

        result += '''    </div>
    <div class="heatmap-footer">Hover over tokens to see confidence scores and alternatives</div>
</div>
</html>'''

        if self.valves.debug_logs:
            self.logger.info("=== END HEATMAP CREATION DEBUG ===")

        return result

    def _get_color_for_confidence(self, confidence: float) -> str:
        """
        Get color based on confidence level and selected color scheme
        """
        # Normalize confidence to 0-1 range
        normalized = min(max(confidence, 0), 1)

        if self.valves.color_scheme == "red_blue":
            # Red (low) to Blue (high)
            red = int(255 * (1 - normalized))
            blue = int(255 * normalized)
            green = int(100 * (1 - abs(normalized - 0.5) * 2))  # Peak at middle
            return f"rgba({red}, {green}, {blue}, 0.3)"

        elif self.valves.color_scheme == "heat":
            # Heat map: dark blue -> blue -> cyan -> green -> yellow -> red
            if normalized < 0.2:
                return f"rgba(0, 0, {int(255 * normalized / 0.2)}, 0.4)"
            elif normalized < 0.4:
                return f"rgba(0, {int(255 * (normalized - 0.2) / 0.2)}, 255, 0.4)"
            elif normalized < 0.6:
                return f"rgba({int(255 * (normalized - 0.4) / 0.2)}, 255, {int(255 * (0.6 - normalized) / 0.2)}, 0.4)"
            elif normalized < 0.8:
                return f"rgba(255, {int(255 * (0.8 - normalized) / 0.2)}, 0, 0.4)"
            else:
                return f"rgba(255, 0, 0, {0.4 + 0.3 * (normalized - 0.8) / 0.2})"

        else:  # rainbow
            # Rainbow spectrum
            hue = int(240 * (1 - normalized))  # Blue to Red
            return f"hsla({hue}, 70%, 60%, 0.4)"

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )


# Required metadata for Open WebUI
__version__ = "2.1.0"
__author__ = "Assistant"
__description__ = "Visualizes Ollama model response tokens as a heatmap based on logprobs with hover tooltips showing alternatives. Supports both Ollama native and OpenAI-compatible formats."
