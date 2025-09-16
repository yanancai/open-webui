"""
Open WebUI Filter Extension: Token Probability Interactive Heatmap Generator
This Filter Function captures model response tokens and logprobs data and generates 
interactive HTML artifacts for visualization.

Compatible with all LLM providers through Open WebUI's unified OpenAI-compatible format.

Input format:
- OpenAI-compatible: {"logprobs": true, "top_logprobs": 5}

Output format:
- OpenAI-compatible: logprobs in choices[0].logprobs.content[] (streaming) or message.logprobs (final)

Focus: Generates interactive HTML artifacts with token probability heatmaps and confidence analysis.
"""

from pydantic import BaseModel, Field
import json
import math
import re
import time
from typing import Optional, List, Dict, Any
import logging


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority level for this filter")
        top_k: int = Field(
            default=5, 
            description="Number of top alternative tokens to capture for heatmap visualization (0-20)",
            ge=0,
            le=20
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True # IMPORTANT: This creates a switch UI in Open WebUI
        # TIP: Use SVG Data URI!
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="""
        
        # Set up logging without interfering with Open WebUI's configuration
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Only set level if not already configured to avoid conflicts
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
        
        # State tracking for streaming responses
        self.streaming_state = {}
        self.processed_messages = set()  # Track which messages have been processed to avoid duplicates
        self.current_chat_id = None  # Track current conversation
        self.conversation_turn_count = 0  # Track conversation turns for refresh detection

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Modify the request to ensure logprobs are requested if toggle is enabled.
        Open WebUI always uses OpenAI-compatible format for filter functions.
        """
        # Immediately return if disabled - no processing at all
        if not self.toggle:
            return body
        
        try:
            # Check for new conversation turn and refresh state if needed
            self._detect_and_handle_conversation_turn(body)
            
            # Periodic resource cleanup to prevent memory leaks
            if hasattr(self, '_last_cleanup'):
                if time.time() - self._last_cleanup > 60:  # Cleanup every minute
                    self._cleanup_resources()
                    self._last_cleanup = time.time()
            else:
                self._last_cleanup = time.time()

            # Open WebUI uses OpenAI-compatible format for filter functions
            # Add logprobs parameters at the top level
            if "logprobs" not in body:
                body["logprobs"] = True
                body["top_logprobs"] = min(max(self.valves.top_k, 0), 20)
        except Exception as e:
            # If anything fails, just return the original body
            pass

        return body

    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process the response and add heatmap visualization if logprobs are present.
        Only processes NEW messages to prevent duplicate artifacts.
        Open WebUI provides responses in a normalized format with logprobs already extracted.
        """
        # Immediately return if disabled - no processing at all
        if not self.toggle:
            return body

        try:
            # Handle the chat completion response structure
            # In this case, we have: {"messages": [...], "chat_id": "...", ...}
            if "messages" in body and isinstance(body["messages"], list):
                
                # Get chat context for artifact refresh logic
                chat_id = body.get("chat_id", "unknown")
                # Reduce verbose logging to prevent performance issues
                if len(self.processed_messages) % 10 == 0:  # Log every 10 messages
                    print(f"🔍 MESSAGE MANAGEMENT - Processing chat: {chat_id}")
                    print(f"🔍 Current conversation state: chat_id={self.current_chat_id}, turn={self.conversation_turn_count}")
                    print(f"🔍 Processed messages count: {len(self.processed_messages)}")
                
                # Find assistant messages with logprobs, but only process NEW ones (not already processed)
                new_assistant_messages = []
                for message in body["messages"]:
                    if message.get("role") == "assistant":
                        message_id = message.get("id", "unknown")
                        # Check if this message was already processed (any version)
                        base_message_key = f"{chat_id}:{message_id}"
                        already_processed = any(key.startswith(base_message_key) for key in self.processed_messages)
                        
                        # Reduce verbose logging to prevent performance issues
                        if not already_processed:
                            print(f"🆔 FOUND new assistant message - ID: {message_id}")
                            print(f"📊 Logprobs present: {bool(message.get('logprobs'))}")
                        
                        if not already_processed:
                            new_assistant_messages.append(message)
                        else:
                            print(f"⚠️ SKIPPING already processed message: {message_id}")
                
                # Process only the NEW assistant messages (typically just one per turn)
                for message in new_assistant_messages:
                    content = message.get("content", "")
                    logprobs_data = message.get("logprobs")
                    message_id = message.get("id", "unknown")
                    
                    if logprobs_data and content.strip():
                        # Create a unique key for this specific message
                        unique_message_key = f"{chat_id}:{message_id}:processed"
                        
                        print(f"🔑 PROCESSING NEW MESSAGE: {message_id}")
                        print(f"📝 Content length: {len(content)} chars")
                        print(f"📊 Logprobs tokens: {len(logprobs_data) if logprobs_data else 0}")
                        
                        # Mark this specific message as processed
                        self.processed_messages.add(unique_message_key)
                        print(f"✅ MARKED AS PROCESSED: {unique_message_key}")
                        
                        # Generate heatmap HTML for this new message
                        heatmap_html = self._create_heatmap_html(content, logprobs_data, self.conversation_turn_count)
                        
                        if heatmap_html:
                            # Append the heatmap to the message content
                            message["content"] = content + "\n\n" + heatmap_html
                            print(f"✅ ARTIFACT GENERATED for NEW message {message_id} (turn {self.conversation_turn_count})")
                        
                        # Keep logprobs for UI access
                        if not message.get("logprobs"):
                            message["logprobs"] = logprobs_data
                        
                        # Process only the first new message to avoid multiple artifacts per turn
                        break
                    elif not logprobs_data:
                        print(f"⚠️ No logprobs found in NEW message {message_id}")
                    elif not content.strip():
                        print(f"⚠️ NEW message {message_id} has logprobs but no content")
                
                # Clean up processed messages more aggressively to prevent memory leaks
                if len(self.processed_messages) > 100:
                    # Keep only the most recent 50 processed messages
                    recent_messages = list(self.processed_messages)[-50:]
                    self.processed_messages = set(recent_messages)
                    print(f"🧹 Cleaned up processed messages cache: {len(self.processed_messages)} entries remaining")
                
                if not new_assistant_messages:
                    print("⚠️ No NEW assistant messages with logprobs found")
                
                return body

            # Handle other response structures if needed
            print("⚠️ Unknown response structure, returning original")

        except Exception as e:
            # Log error but don't let it break the response
            print(f"❌ Error processing logprobs: {e}")

        return body

    def stream(self, event: dict) -> dict:
        """
        Process streaming chunks in real-time to build incremental heatmap visualization.
        Open WebUI provides streaming data in OpenAI-compatible format.
        """
        # Immediately return if disabled - no processing at all
        if not self.toggle:
            return event
            
        try:
            # Get chat_id for state tracking
            chat_id = event.get("id", "unknown")
            
            # Track conversation changes
            if self.current_chat_id != chat_id:
                print(f"🔄 STREAM CONVERSATION CHANGE: {self.current_chat_id} -> {chat_id}")
                self.current_chat_id = chat_id
                self.conversation_turn_count += 1
            
            # Initialize state for this chat if needed
            if chat_id not in self.streaming_state:
                self.streaming_state[chat_id] = {
                    "tokens": [],
                    "logprobs": [],
                    "top_logprobs": [],
                    "content_so_far": "",
                    "last_heatmap": "",
                    "chunk_count": 0,
                    "logprob_chunks": 0,
                    "start_time": None
                }
                
                # Import time for tracking
                import time
                self.streaming_state[chat_id]["start_time"] = time.time()
            
            state = self.streaming_state[chat_id]
            state["chunk_count"] += 1
            
            # Process choices in the streaming event
            choices = event.get("choices", [])
            
            for choice_idx, choice in enumerate(choices):
                delta = choice.get("delta", {})
                
                # Handle content updates
                if "content" in delta:
                    content_chunk = delta["content"]
                    state["content_so_far"] += content_chunk
                
                # Handle logprobs updates (OpenAI format)
                logprobs = delta.get("logprobs")
                if logprobs:
                    state["logprob_chunks"] += 1
                    
                    # Handle OpenAI format: logprobs.content is an array
                    if isinstance(logprobs, dict) and "content" in logprobs:
                        content_logprobs = logprobs["content"]
                        
                        if isinstance(content_logprobs, list):
                            for token_idx, token_logprob in enumerate(content_logprobs):
                                if isinstance(token_logprob, dict):
                                    token = token_logprob.get("token", "")
                                    logprob = token_logprob.get("logprob", None)
                                    top_logprobs = token_logprob.get("top_logprobs", [])
                                    
                                    state["tokens"].append(token)
                                    state["logprobs"].append(logprob)
                                    state["top_logprobs"].append(top_logprobs)
                
                # Check if this is the final chunk
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    print(f"🏁 STREAM FINISHED for chat {chat_id} - turn {self.conversation_turn_count}")
                    
                    # Clean up the streaming state immediately
                    if chat_id in self.streaming_state:
                        del self.streaming_state[chat_id]
                    
                    # Clean up old processed messages more aggressively (keep only recent ones)
                    if len(self.processed_messages) > 150:
                        recent_messages = list(self.processed_messages)[-75:]
                        self.processed_messages = set(recent_messages)
                        print(f"🧹 Stream cleanup: processed messages cache reduced to {len(self.processed_messages)} entries")
                        
        except Exception as e:
            # Log error but don't let it break streaming
            print(f"❌ ERROR in stream processing: {e}")
            
        return event

    def _detect_and_handle_conversation_turn(self, body: dict) -> None:
        """
        Detect new conversation turns and refresh artifact state accordingly
        """
        # Extract chat identifier from the request
        chat_id = None
        
        # Try to get chat_id from various possible locations
        if "chat_id" in body:
            chat_id = body["chat_id"]
        elif "id" in body:
            chat_id = body["id"]
        elif "messages" in body and isinstance(body["messages"], list):
            # Look for chat context in messages
            for msg in body["messages"]:
                if isinstance(msg, dict) and "chat_id" in msg:
                    chat_id = msg["chat_id"]
                    break
        
        print(f"🔍 CONVERSATION TURN DETECTION:")
        print(f"   - Detected chat_id: {chat_id}")
        print(f"   - Current tracked chat_id: {self.current_chat_id}")
        # Only log turn count if it's a new conversation or significant milestone
        if chat_id != self.current_chat_id or self.conversation_turn_count % 5 == 0:
            print(f"   - Current turn count: {self.conversation_turn_count}")
        
        # Check if this is a new conversation or a new turn
        if chat_id and chat_id != self.current_chat_id:
            print(f"🔄 NEW CONVERSATION DETECTED!")
            print(f"   - Previous: {self.current_chat_id}")
            print(f"   - New: {chat_id}")
            
            # Refresh state for new conversation
            self._refresh_conversation_state(chat_id)
            
        elif chat_id:
            # Same conversation, increment turn count for artifact versioning
            self.conversation_turn_count += 1
            # Only log every few turns to reduce verbosity
            if self.conversation_turn_count % 3 == 0:
                print(f"�️ CONTINUING CONVERSATION {chat_id} - Turn {self.conversation_turn_count}")
        else:
            print(f"⚠️ No chat_id detected in request body")
            print(f"📦 Request body keys: {list(body.keys())}")
            # For requests without chat_id, still try to clean some artifacts to be safe
            if len(self.processed_messages) > 0:
                print(f"🧹 Clearing some processed messages as fallback")
                # Keep only half to be conservative
                recent_messages = list(self.processed_messages)[-50:]
                self.processed_messages = set(recent_messages)

    def _refresh_conversation_state(self, new_chat_id: str) -> None:
        """
        Refresh all state when a new conversation is detected
        """
        print(f"🔄 REFRESHING CONVERSATION STATE for chat: {new_chat_id}")
        
        # Update tracking
        old_chat_id = self.current_chat_id
        self.current_chat_id = new_chat_id
        self.conversation_turn_count = 1  # Reset turn count for new conversation
        
        # Clear processed messages from previous conversation
        if old_chat_id:
            # Remove messages from the old conversation to allow fresh artifacts
            old_messages = {msg_key for msg_key in self.processed_messages if msg_key.startswith(f"{old_chat_id}:")}
            self.processed_messages -= old_messages
            print(f"🧹 Cleared {len(old_messages)} processed messages from previous conversation")
        
        # Clear any streaming state from previous conversation
        if old_chat_id in self.streaming_state:
            del self.streaming_state[old_chat_id]
            print(f"🧹 Cleared streaming state from previous conversation")
        
        print(f"✅ State refreshed - ready for new conversation {new_chat_id}")

    def _cleanup_resources(self) -> None:
        """
        Perform aggressive cleanup to prevent memory leaks and resource issues
        """
        try:
            # Clean up old streaming states
            current_time = time.time()
            stale_chats = []
            
            for chat_id, state in self.streaming_state.items():
                # Remove streams older than 5 minutes
                if state.get("start_time") and (current_time - state["start_time"]) > 300:
                    stale_chats.append(chat_id)
            
            for chat_id in stale_chats:
                del self.streaming_state[chat_id]
                print(f"🧹 Removed stale streaming state for chat: {chat_id}")
            
            # Keep only the most recent 25 processed messages to prevent unbounded growth
            if len(self.processed_messages) > 50:
                recent_messages = list(self.processed_messages)[-25:]
                self.processed_messages = set(recent_messages)
                print(f"🧹 Aggressive cleanup: reduced to {len(self.processed_messages)} processed messages")
                
        except Exception as e:
            print(f"⚠️ Error during resource cleanup: {e}")

    def _create_heatmap_html(self, content: str, logprobs_data: list, turn: int) -> str:
        """
        Create HTML with heatmap styling based on OpenAI-compatible logprobs format with turn-based versioning.
        In Open WebUI's outlet, logprobs_data is already extracted and normalized to a consistent format.
        """
        if not isinstance(logprobs_data, list) or not logprobs_data:
            print("Invalid logprobs data format")
            return content

        tokens = []
        token_logprobs = []
        top_logprobs = []

        for item in logprobs_data:
            if isinstance(item, dict):
                tokens.append(item.get("token", ""))
                token_logprobs.append(item.get("logprob", None))
                top_logprobs.append(item.get("top_logprobs", []))

        return self._generate_heatmap_html(tokens, token_logprobs, top_logprobs, turn)

    def _generate_heatmap_html(self, tokens: list, token_logprobs: list, top_logprobs: list, turn: int) -> str:
        """
        Generate HTML code block that Open WebUI will render as an artifact with turn-based versioning
        """

        # Validate that we have matching data
        if len(tokens) != len(token_logprobs):
            print(f"Token count mismatch: {len(tokens)} tokens vs {len(token_logprobs)} logprobs")
            # Truncate to minimum length
            min_len = min(len(tokens), len(token_logprobs))
            tokens = tokens[:min_len]
            token_logprobs = token_logprobs[:min_len]
            top_logprobs = top_logprobs[:min_len] if len(top_logprobs) > min_len else top_logprobs

        if not tokens:
            return ""

        # Calculate probability ranges for color mapping
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if not valid_logprobs:
            return ""
            
        min_logprob = min(valid_logprobs)
        max_logprob = max(valid_logprobs)
        logprob_range = max_logprob - min_logprob if max_logprob != min_logprob else 1

        # Build HTML content for the artifact
        html_content = []
        
        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
            if logprob is not None:
                # Get ranking-based color instead of probability-based
                rank_color = self._get_rank_based_color(i, top_logprobs[i] if i < len(top_logprobs) else [], token)
                
                # Calculate actual probability for tooltip
                probability = math.exp(logprob)
                
                # Get top alternatives for tooltip
                alternatives = ""
                if i < len(top_logprobs) and top_logprobs[i]:
                    alt_list = []
                    for alt in top_logprobs[i]:  # Show all top_k tokens
                        if isinstance(alt, dict):
                            alt_token = alt.get("token", "")
                            alt_logprob = alt.get("logprob", 0)
                            alt_prob = math.exp(alt_logprob)
                            alt_list.append(f"'{alt_token}' ({alt_prob:.1%})")
                    if alt_list:
                        alternatives = f" | Alternatives: {', '.join(alt_list)}"
                
                # Escape token for HTML
                escaped_token = self._escape_html(token)
                
                # Create span with color and tooltip
                token_span = f'<span class="token-span" style="background-color: {rank_color}; padding: 2px 1px; border-radius: 3px;" data-title="Token: {escaped_token} | Probability: {probability:.1%} | Logprob: {logprob:.3f}{alternatives}">{escaped_token}</span>'
            else:
                # No logprob data - render without color
                escaped_token = self._escape_html(token)
                token_span = f'<span class="token-span" data-title="No probability data">{escaped_token}</span>'
            
            html_content.append(token_span)

        # Add unique artifact ID with turn-based versioning for Open WebUI artifact system
        artifact_id = f"logprobs-heatmap-turn{turn}-{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Create the full HTML artifact with turn-based versioning
        artifact_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Probability Heatmap - Turn {turn}</title>
    <!-- Conversation Turn: {turn} | Generated: {timestamp} -->
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            width: 90%;
            max-width: none;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e1e8ed;
            padding-bottom: 20px;
        }}
        .title {{
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin: 10px 0;
            font-size: 1.1rem;
        }}
        .turn-info {{
            background: #e8f5e8;
            border-radius: 5px;
            padding: 8px;
            margin: 10px 0;
            font-size: 0.9rem;
            color: #2d5a2d;
            border-left: 4px solid #4caf50;
        }}
        .text-content {{
            background: #fafbfc;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            padding: 25px;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            line-height: 1.8;
            word-wrap: break-word;
            white-space: pre-wrap;
            margin: 20px 0;
            position: relative;
        }}
        .token-span {{
            cursor: help;
            transition: all 0.2s ease;
            position: relative;
        }}
        .token-span:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .popup {{
            position: absolute;
            background: #2c3e50;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 13px;
            font-family: 'Courier New', monospace;
            white-space: nowrap;
            z-index: 1000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease;
            max-width: 300px;
            line-height: 1.4;
        }}
        .popup.show {{
            opacity: 1;
        }}
        .popup-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 4px 0;
            padding: 3px 6px;
            border-radius: 4px;
        }}
        .popup-row.selected {{
            background: rgba(52, 152, 219, 0.3);
            border: 1px solid #3498db;
        }}
        .popup-row:not(.selected) {{
            background: rgba(255,255,255,0.05);
        }}
        .popup-token {{
            font-family: 'Courier New', monospace;
            margin-right: 10px;
        }}
        .popup-prob {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e1e8ed;
            color: #7f8c8d;
        }}
    </style>
</head>
<body data-artifact-id="{artifact_id}">
    <div class="container">
        <div class="header">
            <h1 class="title">🎯 Token Probability Heatmap</h1>
            <p class="subtitle">💡 Hover over tokens to see detailed probability information and alternatives</p>
            <div class="turn-info">
                📦 Conversation Turn: {turn} | Generated: {timestamp}
            </div>
        </div>

        <div class="text-content" id="text-content">{"".join(html_content)}</div>
        
        <div class="footer">
            📊 **Stats:** {len(tokens)} tokens analyzed with ranking-based confidence visualization
        </div>
    </div>

    <script>
        // Popup functionality
        const popup = document.createElement('div');
        popup.className = 'popup';
        document.body.appendChild(popup);

        document.querySelectorAll('.token-span').forEach(span => {{
            span.addEventListener('mouseenter', function(e) {{
                const rect = this.getBoundingClientRect();
                const title = this.getAttribute('data-title');
                
                if (title) {{
                    // Parse the title to extract token info
                    const tokenMatch = title.match(/Token: ([^|]+)/);
                    const probMatch = title.match(/Probability: ([^|]+)/);
                    const altMatch = title.match(/Alternatives: (.+)/);
                    
                    let content = '';
                    let selectedToken = '';
                    
                    if (tokenMatch) {{
                        selectedToken = tokenMatch[1].trim();
                    }}
                    
                    // Parse all alternative tokens (this includes the selected token in its ranked position)
                    if (altMatch) {{
                        const alternatives = altMatch[1].split(', ');
                        alternatives.forEach(alt => {{
                            const altTokenMatch = alt.match(/'([^']+)' \\(([^)]+)\\)/);
                            if (altTokenMatch) {{
                                const token = altTokenMatch[1];
                                const prob = altTokenMatch[2];
                                const isSelected = token === selectedToken;
                                const cssClass = isSelected ? 'popup-row selected' : 'popup-row';
                                
                                content += `<div class="${{cssClass}}">`;
                                content += `<span class="popup-token">${{token}}</span>`;
                                content += `<span class="popup-prob">${{prob}}</span>`;
                                content += `</div>`;
                            }}
                        }});
                    }}
                    
                    popup.innerHTML = content;
                    popup.style.left = rect.left + 'px';
                    popup.style.top = (rect.top - popup.offsetHeight - 10) + 'px';
                    popup.classList.add('show');
                }}
            }});
            
            span.addEventListener('mouseleave', function() {{
                popup.classList.remove('show');
            }});
        }});

        // Log artifact turn for debugging
        console.log('Logprobs Heatmap Artifact Turn {turn} loaded at {timestamp}');
    </script>
</body>
</html>'''

        # Return as a code block that Open WebUI will detect and render as an artifact
        result = f"\n\n🎯 **Token Probability Heatmap Generated!** (Conversation Turn {turn})\n\n```html\n{artifact_html}\n```\n\n📊 **Stats:** {len(tokens)} tokens analyzed with confidence visualization | Artifact ID: `{artifact_id}`\n"
        
        return result

    def _get_rank_based_color(self, token_index: int, top_logprobs_list: list, current_token: str) -> str:
        """
        Get color based on the ranking of the current token in the top_logprobs list
        """
        if not top_logprobs_list:
            # Default color when no alternatives available
            return "rgb(200, 200, 200)"
        
        # Find the rank of current token in the alternatives list
        rank = 0  # Default to rank 0 (highest probability)
        
        for i, alt in enumerate(top_logprobs_list):
            if isinstance(alt, dict) and alt.get("token", "") == current_token:
                rank = i
                break
        
        # Total number of alternatives including the current token
        total_alternatives = len(top_logprobs_list)
        
        # Calculate color based on rank (0 = best rank = green, higher rank = more red)
        if total_alternatives <= 1:
            # Only one option, make it green
            return "rgb(100, 255, 100)"
        
        # Normalize rank to 0-1 range
        normalized_rank = rank / (total_alternatives - 1)
        
        # Create color gradient from green (rank 0) to red (highest rank)
        red = int(255 * normalized_rank)
        green = int(255 * (1 - normalized_rank))
        blue = 100
        
        return f"rgb({red}, {green}, {blue})"

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
__version__ = "7.1.0"
__author__ = "Assistant"
__description__ = "Captures model response tokens and logprobs data to generate interactive HTML artifacts. Creates beautiful, hoverable heatmap visualizations with confidence analysis and detailed statistics. Features NEW MESSAGE ONLY processing to prevent duplicate artifacts. Each conversation turn gets its own unique artifact. Works with all LLM providers through Open WebUI's unified OpenAI-compatible format."
