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
- OpenAI-compatible: {"logprobs": true, "top_logprob        # MINIMAL TEST - create proper HTML        # MINIMAL TEST - clean code blocks for Artifacts
        token_count = len(logprobs_data) if isinstance(logprobs_data, list) else 0

        # Return clean code blocks without extra markdown
        result = f"""```html
<div class="test-container">
    <h1>Logprobs Test</h1>
    <p>Original: {self._escape_html(content)}</p>
    <p>Tokens: {token_count}</p>
    <div class="success">Artifacts Working!</div>
</div>
```

```css
.test-container {{
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    text-align: center;
    font-family: Arial, sans-serif;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.test-container h1 {{
    font-size: 2em;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}}

.test-container p {{
    font-size: 1.2em;
    margin: 10px 0;
}}

.success {{
    background: rgba(255,255,255,0.2);
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    font-weight: bold;
    font-size: 1.3em;
}}
```"""
        token_count = len(logprobs_data) if isinstance(logprobs_data, list) else 0

        # Return with separate code blocks that Artifacts will combine
        result = f"""**Original Response:** {content}

**Logprobs Found:** {token_count} tokens

```html
<div class="logprobs-debug">
    <h1>RED LOGPROBS DEBUG TEST</h1>
    <p><strong>Original:</strong> {self._escape_html(content)}</p>
    <p><strong>Tokens Found:</strong> {token_count}</p>
    <div class="status">Artifacts System Working!</div>
</div>
```

```css
.logprobs-debug {{
    font-family: Arial, sans-serif;
    padding: 20px;
    border: 3px solid red;
    background: #ffe6e6;
    border-radius: 10px;
    max-width: 500px;
}}

.logprobs-debug h1 {{
    color: red;
    margin: 0 0 15px 0;
}}

.logprobs-debug p {{
    margin: 8px 0;
    color: #333;
}}

.status {{
    background: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin-top: 15px;
    font-weight: bold;
}}
```"""formats supported:
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
            default=False, description="Enable logprob heatmap visualization (disable for debugging raw logprobs)"
        )
        preserve_logprobs: bool = Field(
            default=True, description="Always preserve raw logprobs data in message object for UI access"
        )
        show_debug_ui: bool = Field(
            default=True, description="Show debug UI with raw logprobs data even when heatmap is disabled"
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
                                message["content"] = heatmap_html
                                
                                if self.valves.debug_logs:
                                    self.logger.info("✅ Successfully replaced message content with heatmap")
                                    self.logger.info(f"Original content: '{content}'")
                                    self.logger.info(f"New content length: {len(heatmap_html)} characters")
                            elif self.valves.show_debug_ui:
                                debug_html = self._create_debug_ui_html(content, logprobs_data)
                                message["content"] = debug_html
                                
                                if self.valves.debug_logs:
                                    self.logger.info("✅ Successfully replaced message content with debug UI")
                                    self.logger.info(f"Original content: '{content}'")
                                    self.logger.info(f"New content length: {len(debug_html)} characters")
                            
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
            # Handle Ollama's native API response structure
            # For /api/chat endpoint, logprobs are in message.logprobs
            # For /api/generate endpoint, logprobs are in response.logprobs
            
            # Check for streaming response - but still process logprobs if available
            is_streaming_chunk = "delta" in body
            
            if is_streaming_chunk and self.valves.debug_logs:
                self.logger.info("Processing streaming chunk response")
            
            # Handle /api/chat response structure
            if "message" in body:
                message = body["message"]
                content = message.get("content", "")
                logprobs_data = message.get("logprobs")
                
                if self.valves.debug_logs:
                    self.logger.info("Processing Ollama /api/chat response")
                    self.logger.info(f"Message: {json.dumps(message, indent=2, default=str)}")
                
                if logprobs_data:
                    if self.valves.debug_logs:
                        self.logger.info("Found logprobs in chat message!")
                        self.logger.info(f"Logprobs data: {json.dumps(logprobs_data, indent=2, default=str)}")
                    
                    # Store the logprobs data in the message for UI access
                    # This ensures the raw logprobs are available even if we modify content
                    if self.valves.preserve_logprobs and not message.get("logprobs"):
                        message["logprobs"] = logprobs_data
                        if self.valves.debug_logs:
                            self.logger.info("✅ Added logprobs data to message for UI access")
                    
                    # Only process if we have actual content and this is the final response
                    # For streaming, we need to accumulate content
                    if content.strip() and not is_streaming_chunk:
                        if self.valves.enable_heatmap:
                            heatmap_html = self._create_heatmap_html_ollama(content, logprobs_data)
                            message["content"] = heatmap_html
                            
                            if self.valves.debug_logs:
                                self.logger.info("✅ Successfully replaced chat message content with heatmap")
                                self.logger.info(f"Original content: '{content}'")
                                self.logger.info(f"New content length: {len(heatmap_html)} characters")
                        elif self.valves.show_debug_ui:
                            debug_html = self._create_debug_ui_html(content, logprobs_data)
                            message["content"] = debug_html
                            
                            if self.valves.debug_logs:
                                self.logger.info("✅ Successfully added debug UI to chat message content")
                                self.logger.info(f"Original content: '{content}'")
                                self.logger.info(f"New content length: {len(debug_html)} characters")
                        else:
                            if self.valves.debug_logs:
                                self.logger.info("Both heatmap and debug UI disabled - keeping original content but preserving logprobs data")
                    elif is_streaming_chunk:
                        if self.valves.debug_logs:
                            self.logger.info("Streaming chunk detected - preserving logprobs but skipping heatmap for this chunk")
                else:
                    if self.valves.debug_logs:
                        self.logger.warning("No logprobs found in chat message")
                
                return body
            
            # Handle /api/generate response structure
            elif "response" in body:
                content = body.get("response", "")
                logprobs_data = body.get("logprobs")
                
                if self.valves.debug_logs:
                    self.logger.info("Processing Ollama /api/generate response")
                    self.logger.info(f"Response keys: {list(body.keys())}")
                
                if logprobs_data:
                    if self.valves.debug_logs:
                        self.logger.info("Found logprobs in generate response!")
                        self.logger.info(f"Logprobs data: {json.dumps(logprobs_data, indent=2, default=str)}")
                    
                    # Store logprobs in the response body for UI access
                    if self.valves.preserve_logprobs and not body.get("logprobs"):
                        body["logprobs"] = logprobs_data
                        if self.valves.debug_logs:
                            self.logger.info("✅ Added logprobs data to response body for UI access")
                    
                    if self.valves.enable_heatmap:
                        heatmap_html = self._create_heatmap_html_ollama(content, logprobs_data)
                        body["response"] = heatmap_html
                        
                        if self.valves.debug_logs:
                            self.logger.info("✅ Successfully replaced generate response content with heatmap")
                            self.logger.info(f"Original content: '{content}'")
                            self.logger.info(f"New content length: {len(heatmap_html)} characters")
                    elif self.valves.show_debug_ui:
                        debug_html = self._create_debug_ui_html(content, logprobs_data)
                        body["response"] = debug_html
                        
                        if self.valves.debug_logs:
                            self.logger.info("✅ Successfully added debug UI to generate response content")
                            self.logger.info(f"Original content: '{content}'")
                            self.logger.info(f"New content length: {len(debug_html)} characters")
                    else:
                        if self.valves.debug_logs:
                            self.logger.info("Both heatmap and debug UI disabled - keeping original content but preserving logprobs data")
                else:
                    if self.valves.debug_logs:
                        self.logger.warning("No logprobs found in generate response")
                
                return body
            
            # Handle OpenAI-compatible endpoint response
            choices = body.get("choices", [])
            if choices:
                choice = choices[0]
                
                if self.valves.debug_logs:
                    self.logger.info("Processing OpenAI-compatible response")
                    self.logger.info(f"Choice structure: {json.dumps(choice, indent=2, default=str)}")

                # For streaming responses, check if this chunk has logprobs
                if "delta" in choice:
                    # This is a streaming chunk
                    delta = choice.get("delta", {})
                    logprobs_data = choice.get("logprobs")

                    if self.valves.debug_logs:
                        self.logger.info(f"Processing streaming delta: {json.dumps(delta, indent=2, default=str)}")
                        self.logger.info(f"Delta logprobs: {json.dumps(logprobs_data, indent=2, default=str) if logprobs_data else 'None'}")

                    if logprobs_data:
                        content = delta.get("content", "")
                        
                        # Store logprobs in the choice for UI access
                        if self.valves.preserve_logprobs and not choice.get("logprobs_data"):
                            choice["logprobs_data"] = logprobs_data
                            if self.valves.debug_logs:
                                self.logger.info("✅ Added logprobs data to choice for UI access")
                        
                        # Only process if we have content - streaming chunks may be empty
                        if content.strip() and self.valves.enable_heatmap:
                            heatmap_html = self._create_heatmap_html_openai(content, logprobs_data)
                            delta["content"] = heatmap_html
                            
                            if self.valves.debug_logs:
                                self.logger.info("✅ Successfully processed OpenAI streaming chunk with logprobs")
                        else:
                            if self.valves.debug_logs:
                                self.logger.info("Streaming chunk has logprobs but no content - keeping as is")

                    return body
                else:
                    # Non-streaming response
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    logprobs_data = choice.get("logprobs")

                    if self.valves.debug_logs:
                        self.logger.info(f"Processing non-streaming message: {json.dumps(message, indent=2, default=str)}")
                        self.logger.info(f"Choice logprobs: {json.dumps(logprobs_data, indent=2, default=str) if logprobs_data else 'None'}")

                    if logprobs_data and content.strip():
                        # Store logprobs in the message for UI access
                        if self.valves.preserve_logprobs and not message.get("logprobs"):
                            message["logprobs"] = logprobs_data
                            if self.valves.debug_logs:
                                self.logger.info("✅ Added logprobs data to message for UI access")
                        
                        if self.valves.enable_heatmap:
                            heatmap_html = self._create_heatmap_html_openai(content, logprobs_data)
                            message["content"] = heatmap_html
                            
                            if self.valves.debug_logs:
                                self.logger.info("✅ Successfully processed OpenAI non-streaming response with logprobs")
                        elif self.valves.show_debug_ui:
                            debug_html = self._create_debug_ui_html(content, logprobs_data, is_openai=True)
                            message["content"] = debug_html
                            
                            if self.valves.debug_logs:
                                self.logger.info("✅ Successfully added debug UI to OpenAI non-streaming response")
                        else:
                            if self.valves.debug_logs:
                                self.logger.info("Both heatmap and debug UI disabled - keeping original content but preserving logprobs data")
                    elif not logprobs_data:
                        if self.valves.debug_logs:
                            self.logger.warning("No logprobs found in OpenAI response")

                    return body
            
            if self.valves.debug_logs:
                self.logger.warning("Unknown response structure, returning original")

        except Exception as e:
            if self.valves.debug_logs:
                self.logger.error(f"Error processing logprobs: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")

        if self.valves.debug_logs:
            # Check if any content was modified
            modified = False
            heatmap_modified = False
            debug_modified = False
            
            if "message" in body:
                content_str = str(body["message"].get("content", ""))
                if "logprob-heatmap-container" in content_str:
                    heatmap_modified = True
                elif "logprobs-debug-container" in content_str:
                    debug_modified = True
            elif "response" in body:
                content_str = str(body.get("response", ""))
                if "logprob-heatmap-container" in content_str:
                    heatmap_modified = True
                elif "logprobs-debug-container" in content_str:
                    debug_modified = True
            elif "choices" in body:
                for choice in body["choices"]:
                    if "message" in choice:
                        content_str = str(choice["message"].get("content", ""))
                        if "logprob-heatmap-container" in content_str:
                            heatmap_modified = True
                            break
                        elif "logprobs-debug-container" in content_str:
                            debug_modified = True
                            break
            
            if heatmap_modified:
                self.logger.info("🎨 Response content successfully transformed with logprobs heatmap")
            elif debug_modified:
                self.logger.info("🔍 Response content successfully enhanced with logprobs debug UI")
            else:
                self.logger.info("📄 Response returned unchanged (no logprobs found or processing failed)")
            
            self.logger.info("=== END OUTLET DEBUG ===")

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

    def _create_heatmap_html_openai(self, content: str, logprobs_data: dict) -> str:
        """
        Create HTML with heatmap styling based on OpenAI's logprobs format
        """
        if self.valves.debug_logs:
            self.logger.info("=== OPENAI HEATMAP CREATION DEBUG ===")
            self.logger.info(f"Content length: {len(content)}")
            self.logger.info(f"Logprobs data keys: {list(logprobs_data.keys()) if isinstance(logprobs_data, dict) else 'not a dict'}")
            self.logger.info(f"Full logprobs data: {json.dumps(logprobs_data, indent=2, default=str)}")

        # Handle the OpenAI API format for logprobs - content is an array of token objects
        content_logprobs = logprobs_data.get("content", [])

        if not content_logprobs:
            if self.valves.debug_logs:
                self.logger.warning("No content logprobs found in OpenAI format, returning original content")
            return content

        # Extract data from content array (OpenAI format)
        tokens = []
        token_logprobs = []
        top_logprobs = []

        for item in content_logprobs:
            if isinstance(item, dict):
                tokens.append(item.get("token", ""))
                token_logprobs.append(item.get("logprob", None))
                
                # Handle top_logprobs - it's an array of objects with token and logprob
                top_logprobs_item = item.get("top_logprobs", [])
                if isinstance(top_logprobs_item, list):
                    # Convert to the format expected by the heatmap generator
                    formatted_top_logprobs = []
                    for top_item in top_logprobs_item:
                        if isinstance(top_item, dict) and "token" in top_item and "logprob" in top_item:
                            formatted_top_logprobs.append({
                                "token": top_item["token"],
                                "logprob": top_item["logprob"]
                            })
                    top_logprobs.append(formatted_top_logprobs)
                else:
                    top_logprobs.append([])
            else:
                if self.valves.debug_logs:
                    self.logger.warning(f"Invalid content logprob item format: {type(item)}")
                continue

        if self.valves.debug_logs:
            self.logger.info(f"Extracted {len(tokens)} tokens from OpenAI content logprobs")
            self.logger.info(f"First 3 tokens: {tokens[:3]}")
            self.logger.info(f"First 3 logprobs: {token_logprobs[:3]}")
            self.logger.info(f"First top_logprobs sample: {top_logprobs[0] if top_logprobs else 'None'}")

        if not tokens or not token_logprobs:
            if self.valves.debug_logs:
                self.logger.warning("No tokens or logprobs extracted from OpenAI format, returning original content")
            return content

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

        # ULTRA SIMPLE TEST - just wrap original content in red text
        simple_html = f"""
        <div style="color: red; font-weight: bold; border: 2px solid red; padding: 10px;">
            <h1>� LOGPROBS DEBUG TEST - RED TEXT</h1>
            <p>Original content: {self._escape_html(content)}</p>
            <p>Logprobs found: {len(logprobs_data) if isinstance(logprobs_data, list) else 'Not a list'}</p>
        </div>
        """

        # Return with HTML code block for Artifacts
        result = f"""
```html
{simple_html}
```
"""

        if self.valves.debug_logs:
            self.logger.info("=== END SIMPLE DEBUG UI CREATION ===")

        return result

    def _generate_heatmap_html(self, tokens: list, token_logprobs: list, top_logprobs: list) -> str:
        """
        Generate HTML heatmap using code blocks that Open WebUI Artifacts can render
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

        # Generate HTML content
        html_content = '<div class="logprob-heatmap-container">'
        html_content += '<div class="heatmap-header">🔥 Ollama Logprobs Heatmap Visualization</div>'
        html_content += '<div class="heatmap-content">'

        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
            if logprob is None:
                confidence = 0.5  # neutral for unknown
            else:
                # Convert logprob to confidence (probability)
                confidence = math.exp(logprob) if logprob > -10 else 0.01

            # Skip if below threshold
            if confidence < self.valves.min_confidence_threshold:
                html_content += f'<span class="token low-confidence">{self._escape_html(token)}</span>'
                continue

            # Get color based on confidence
            color = self._get_color_for_confidence(confidence)

            # Create hover data for alternatives
            alternatives_data = ""
            if i < len(top_logprobs) and top_logprobs[i]:
                alternatives_list = []

                # Handle different formats for top_logprobs
                top_logprob_item = top_logprobs[i]
                if isinstance(top_logprob_item, list):
                    # Handle both Ollama format (list of dicts) and OpenAI format (list of dicts)
                    for alt in top_logprob_item[: self.valves.show_top_k]:
                        if isinstance(alt, dict):
                            alt_token = alt.get("token", "")
                            alt_logprob = alt.get("logprob", 0)
                            alt_confidence = (
                                math.exp(alt_logprob) * 100
                                if alt_logprob > -10
                                else 0.01
                            )
                            alternatives_list.append(
                                f"'{alt_token}' ({alt_confidence:.1f}%)"
                            )
                elif isinstance(top_logprob_item, dict):
                    # Legacy OpenAI format: dict with token as key and logprob as value
                    for alt_token, alt_logprob in list(top_logprob_item.items())[
                        : self.valves.show_top_k
                    ]:
                        alt_confidence = (
                            math.exp(alt_logprob) * 100 if alt_logprob > -10 else 0.01
                        )
                        alternatives_list.append(
                            f"'{alt_token}' ({alt_confidence:.1f}%)"
                        )

                alternatives_data = ", ".join(alternatives_list)

            html_content += f"""
            <span class="token heatmap-token" 
                  style="background-color: {color};" 
                  data-confidence="{confidence:.3f}"
                  data-logprob="{logprob:.3f}" 
                  data-alternatives="{alternatives_data}">
                {self._escape_html(token)}
            </span>"""

        html_content += '</div>'
        html_content += '<div class="heatmap-footer">Hover over tokens to see alternatives and confidence scores</div>'
        html_content += '</div>'

        # Generate CSS content
        css_content = """
        .logprob-heatmap-container {
            font-family: 'Courier New', monospace;
            line-height: 1.6;
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
            margin: 10px 0;
            position: relative;
            border: 2px solid #007bff;
        }
        
        .heatmap-header {
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .heatmap-content {
            margin: 15px 0;
            line-height: 2;
        }
        
        .heatmap-footer {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            text-align: right;
        }
        
        .token {
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 3px;
            display: inline-block;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }
        
        .token.low-confidence {
            background-color: #e9ecef;
            color: #6c757d;
        }
        
        .token.heatmap-token:hover {
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .token.heatmap-token:hover::after {
            content: attr(data-confidence) "% confidence\\A" "Logprob: " attr(data-logprob) "\\A" "Alternatives: " attr(data-alternatives);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: pre-line;
            z-index: 1000;
            margin-bottom: 5px;
            max-width: 300px;
            pointer-events: none;
        }
        
        .heatmap-legend {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
            padding: 8px;
            background: rgba(255,255,255,0.7);
            border-radius: 4px;
            font-size: 12px;
        }
        
        .legend-gradient {
            width: 200px;
            height: 20px;
            border-radius: 10px;
            margin: 0 10px;
            background: linear-gradient(to right, """ + self._get_legend_gradient() + """);
        }
        """

        # Generate JavaScript content for enhanced interactivity
        js_content = """
        document.addEventListener('DOMContentLoaded', function() {
            // Add legend to heatmap container
            const container = document.querySelector('.logprob-heatmap-container');
            if (container && !container.querySelector('.heatmap-legend')) {
                const legend = document.createElement('div');
                legend.className = 'heatmap-legend';
                legend.innerHTML = 
                    '<span>Low Confidence</span>' +
                    '<div class="legend-gradient"></div>' +
                    '<span>High Confidence</span>';
                container.appendChild(legend);
            }
        });
        """

        # Create the response using code blocks that Artifacts can parse
        result = f"""## 🔥 Logprobs Heatmap Visualization

```html
{html_content}
```

```css
{css_content}
```

```javascript
{js_content}
```"""

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

    def _get_legend_gradient(self) -> str:
        """Get CSS gradient string for legend"""
        if self.valves.color_scheme == "red_blue":
            return "rgba(255,100,100,0.3), rgba(100,100,255,0.3)"
        elif self.valves.color_scheme == "heat":
            return "rgba(0,0,128,0.4), rgba(0,255,255,0.4), rgba(255,255,0,0.4), rgba(255,0,0,0.4)"
        else:  # rainbow
            return "hsla(240,70%,60%,0.4), hsla(180,70%,60%,0.4), hsla(120,70%,60%,0.4), hsla(60,70%,60%,0.4), hsla(0,70%,60%,0.4)"

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
