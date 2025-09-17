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
        self.streaming_processed_chats = set()  # Track chats that were processed during streaming
        self.chat_id_mapping = {}  # Map streaming IDs to final chat IDs
        self.recent_artifacts = {}  # Track recent artifact generation with timestamps

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
        Handles both streaming and non-streaming response formats.
        """
        # Immediately return if disabled - no processing at all
        if not self.toggle:
            return body

        try:
            # Handle the chat completion response structure
            if "messages" in body and isinstance(body["messages"], list):
                
                chat_id = body.get("chat_id", "unknown")
                print(f"🔄 OUTLET processing response for chat {chat_id}")
                print(f"🔍 DEBUG: Found {len(body['messages'])} messages in response")
                print(f"🔍 DEBUG: Response keys: {list(body.keys())}")
                
                # Find assistant messages with logprobs and process them
                found_assistant = False
                for message in body["messages"]:
                    if message.get("role") == "assistant":
                        found_assistant = True
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        logprobs_data = message.get("logprobs")
                        message_id = message.get("id", "unknown")
                        
                        print(f"🔍 DEBUG: Found assistant message {message_id}")
                        print(f"📝 Content length: {len(content)} chars")
                        print(f"📊 Logprobs present: {bool(logprobs_data)}")
                        
                        if logprobs_data:
                            print(f"📊 Logprobs data type: {type(logprobs_data)}")
                            if isinstance(logprobs_data, list):
                                print(f"📊 Logprobs tokens: {len(logprobs_data)}")
                            elif isinstance(logprobs_data, dict):
                                print(f"📊 Logprobs keys: {list(logprobs_data.keys())}")
                        
                        # Check if already processed to avoid duplicates
                        unique_message_key = f"{chat_id}:{message_id}:processed"
                        if unique_message_key in self.processed_messages:
                            print(f"⚠️ SKIPPING already processed message: {message_id}")
                            continue
                        
                        # Check for recent artifact generation to prevent duplicates from streaming
                        current_time = time.time()
                        
                        # Check if this came from streaming (different chat ID patterns)
                        is_likely_from_streaming = any(
                            streaming_id in chat_id or chat_id in streaming_id
                            for streaming_id in self.streaming_processed_chats
                        )
                        
                        if is_likely_from_streaming:
                            print(f"⚠️ SKIPPING - outlet message likely from recent streaming for chat pattern: {chat_id}")
                            continue
                        
                        # Also check timestamp-based recent artifacts
                        turn_key = f"turn{self.conversation_turn_count}"
                        if turn_key in self.recent_artifacts:
                            last_artifact_time = self.recent_artifacts[turn_key]
                            if current_time - last_artifact_time < 5:  # Within last 5 seconds
                                print(f"⚠️ SKIPPING - artifact for turn {self.conversation_turn_count} was generated {current_time - last_artifact_time:.1f}s ago")
                                continue
                        
                        # Check if this message already contains a heatmap artifact to prevent duplicates
                        if "logprobs-heatmap" in content:
                            print(f"⚠️ SKIPPING - message {message_id} already contains heatmap artifact")
                            continue
                        
                        # Check if we have content but no logprobs (logprobs were dropped due to truncation)
                        if content.strip() and not logprobs_data:
                            print(f"📝 CONTENT WITHOUT LOGPROBS detected - text length: {len(content)} chars")
                            print(f"💡 This likely means logprobs were truncated due to token limit")
                            print(f"✅ Displaying text without heatmap for message {message_id}")
                            # Keep the content as-is, no heatmap needed
                            continue
                        
                        # Process if we have logprobs (regardless of content being empty)
                        if logprobs_data:
                            print(f"🔑 PROCESSING MESSAGE: {message_id}")
                            
                            # If content is empty but we have logprobs, reconstruct content from tokens
                            original_content_empty = not content.strip()
                            if original_content_empty and logprobs_data:
                                print(f"📝 Content is empty, reconstructing from logprobs tokens")
                                content = self._reconstruct_content_from_logprobs(logprobs_data)
                                print(f"📝 Reconstructed content length: {len(content)} chars")
                                # Update the message content with reconstructed text
                                message["content"] = content
                            
                            # Additional check: if we still have no content, there's a problem
                            if not content.strip():
                                print(f"❌ ERROR: Still no content after reconstruction attempt")
                                print(f"    - Original content empty: {original_content_empty}")
                                print(f"    - Logprobs type: {type(logprobs_data)}")
                                print(f"    - Logprobs length: {len(logprobs_data) if isinstance(logprobs_data, list) else 'not a list'}")
                                continue
                            
                            # Mark this specific message as processed
                            self.processed_messages.add(unique_message_key)
                            print(f"✅ MARKED AS PROCESSED: {unique_message_key}")
                            
                            # Generate heatmap HTML for this message
                            heatmap_html = self._create_heatmap_html(content, logprobs_data, self.conversation_turn_count)
                            
                            if heatmap_html:
                                # Append the heatmap to the message content
                                message["content"] = content + "\n\n" + heatmap_html
                                print(f"✅ ARTIFACT GENERATED for message {message_id} (turn {self.conversation_turn_count})")
                                
                                # Record artifact generation timestamp to prevent duplicates
                                self.recent_artifacts[f"turn{self.conversation_turn_count}"] = time.time()
                            else:
                                print(f"❌ Failed to generate heatmap HTML for message {message_id}")
                            
                            # Keep logprobs for UI access
                            message["logprobs"] = logprobs_data
                            
                            # Process only the first message with logprobs to avoid multiple artifacts
                            break
                        else:
                            # No logprobs found - this is normal when logprobs are truncated
                            if content.strip():
                                print(f"📝 Message {message_id} has content ({len(content)} chars) but no logprobs - likely truncated")
                            else:
                                print(f"⚠️ Message {message_id} has no content and no logprobs")
                
                if not found_assistant:
                    print(f"⚠️ No assistant messages found in response")
                else:
                    print(f"✅ Found assistant messages, but may have been skipped due to duplicates or missing logprobs")
                
                # Clean up processed messages more aggressively to prevent memory leaks
                if len(self.processed_messages) > 100:
                    # Keep only the most recent 50 processed messages
                    recent_messages = list(self.processed_messages)[-50:]
                    self.processed_messages = set(recent_messages)
                    print(f"🧹 Cleaned up processed messages cache: {len(self.processed_messages)} entries remaining")
                
                return body

            # Handle other response structures if needed
            print("⚠️ Unknown response structure, returning original")

        except Exception as e:
            # Log error but don't let it break the response
            print(f"❌ Error processing logprobs: {e}")

        return body

    def stream(self, event: dict) -> dict:
        """
        Process streaming chunks in real-time to build and update heatmap visualization progressively.
        Each chunk updates the artifact with new tokens as they arrive.
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
                    "chunk_count": 0,
                    "logprob_chunks": 0,
                    "start_time": time.time(),
                    "artifact_generated": False,
                    "last_update_time": 0
                }
                print(f"🆕 INITIALIZED streaming state for chat {chat_id}")
            
            state = self.streaming_state[chat_id]
            state["chunk_count"] += 1
            
            # Process choices in the streaming event
            choices = event.get("choices", [])
            content_updated = False
            logprobs_updated = False
            
            for choice_idx, choice in enumerate(choices):
                delta = choice.get("delta", {})
                
                # Handle content updates
                if "content" in delta:
                    content_chunk = delta["content"]
                    state["content_so_far"] += content_chunk
                    content_updated = True
                
                # Handle logprobs updates - check both delta.logprobs and choice.logprobs
                logprobs = delta.get("logprobs")
                choice_logprobs = choice.get("logprobs")
                
                # Process delta logprobs (typical streaming format)
                if logprobs:
                    state["logprob_chunks"] += 1
                    logprobs_updated = True
                    print(f"[DEBUG] Found logprobs in delta: {logprobs}")
                    
                    # Handle OpenAI format: logprobs.content is an array
                    if isinstance(logprobs, dict) and "content" in logprobs:
                        content_logprobs = logprobs["content"]
                        print(f"[DEBUG] Processing delta content_logprobs: {len(content_logprobs) if isinstance(content_logprobs, list) else 'not a list'}")
                        
                        if isinstance(content_logprobs, list):
                            for token_idx, token_logprob in enumerate(content_logprobs):
                                if isinstance(token_logprob, dict):
                                    token = token_logprob.get("token", "")
                                    logprob = token_logprob.get("logprob", None)
                                    top_logprobs = token_logprob.get("top_logprobs", [])
                                    
                                    state["tokens"].append(token)
                                    state["logprobs"].append(logprob)
                                    state["top_logprobs"].append(top_logprobs)
                                    print(f"[DEBUG] Added token from delta: '{token}' (total: {len(state['tokens'])})")
                
                # Process choice logprobs (alternative format - what we're seeing in the logs)
                elif choice_logprobs:
                    state["logprob_chunks"] += 1
                    logprobs_updated = True
                    print(f"[DEBUG] Found logprobs in choice: {choice_logprobs}")
                    
                    # Handle choice logprobs format: choice.logprobs.content is an array
                    if isinstance(choice_logprobs, dict) and "content" in choice_logprobs:
                        content_logprobs = choice_logprobs["content"]
                        print(f"[DEBUG] Processing choice content_logprobs: {len(content_logprobs) if isinstance(content_logprobs, list) else 'not a list'}")
                        
                        if isinstance(content_logprobs, list):
                            for token_idx, token_logprob in enumerate(content_logprobs):
                                if isinstance(token_logprob, dict):
                                    token = token_logprob.get("token", "")
                                    logprob = token_logprob.get("logprob", None)
                                    top_logprobs = token_logprob.get("top_logprobs", [])
                                    
                                    state["tokens"].append(token)
                                    state["logprobs"].append(logprob)
                                    state["top_logprobs"].append(top_logprobs)
                                    print(f"[DEBUG] Added token from choice: '{token}' (total: {len(state['tokens'])})")
                    else:
                        print(f"[DEBUG] Choice logprobs format not recognized: {type(choice_logprobs)}")
                
                if not logprobs and not choice_logprobs:
                    print(f"[DEBUG] No logprobs found in this chunk (delta or choice)")
                else:
                    print(f"[DEBUG] Logprobs processing complete for this chunk")
                
                # Generate/update heatmap artifact progressively - but don't inject into stream yet
                if logprobs_updated and len(state["tokens"]) > 0:
                    current_time = time.time()
                    # Update internal state every 0.5 seconds or when we have significant new content
                    if (current_time - state["last_update_time"] > 0.5) or len(state["tokens"]) % 3 == 0:
                        # Just track that we should update, don't modify the stream content yet
                        state["last_update_time"] = current_time
                        if len(state["tokens"]) % 5 == 0:  # Log every 5 tokens
                            print(f"🔄 Streaming heatmap ready - {len(state['tokens'])} tokens collected")
                
                # Check if this is the final chunk
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    print(f"🏁 STREAM FINISHED for chat {chat_id} - turn {self.conversation_turn_count}")
                    print(f"[DEBUG] Final streaming state - tokens collected: {len(state['tokens'])}")
                    
                    # Check if we have any tokens with logprobs to generate heatmap
                    if len(state["tokens"]) > 0 and any(lp is not None for lp in state["logprobs"]):
                        print(f"[DEBUG] Generating final streaming heatmap for {len(state['tokens'])} tokens with logprobs")
                        self._finalize_streaming_heatmap(chat_id, state, choice)
                    else:
                        # Content without logprobs - normal case when logprobs are truncated
                        content_length = len(state.get("content_so_far", ""))
                        if content_length > 0:
                            print(f"[DEBUG] Streaming completed with {content_length} chars but no logprobs - likely truncated due to token limit")
                            print(f"[DEBUG] This is normal behavior - text will display without heatmap")
                        else:
                            print(f"[DEBUG] No content or logprobs collected during streaming")
                    
                    # Mark this chat as having been processed during streaming
                    self.streaming_processed_chats.add(chat_id)
                    print(f"📝 MARKED chat {chat_id} as streaming-processed to prevent outlet duplication")
                    
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
                print(f"🔄 CONTINUING CONVERSATION {chat_id} - Turn {self.conversation_turn_count}")
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
        
        # Clear streaming processed markers from previous conversation  
        if old_chat_id:
            self.streaming_processed_chats.discard(old_chat_id)
            print(f"🧹 Cleared streaming processed marker from previous conversation")
        
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
            
            # Clean up old streaming processed chat markers more frequently
            if len(self.streaming_processed_chats) > 10:
                # Keep only the most recent 5
                recent_chats = list(self.streaming_processed_chats)[-5:]
                self.streaming_processed_chats = set(recent_chats)
                print(f"🧹 Cleaned up streaming processed chats: {len(self.streaming_processed_chats)} entries remaining")
            
            # Clean up old artifact timestamps
            if len(self.recent_artifacts) > 50:
                current_time = time.time()
                # Remove artifacts older than 30 seconds
                old_artifacts = [key for key, timestamp in self.recent_artifacts.items() 
                               if current_time - timestamp > 30]
                for key in old_artifacts:
                    del self.recent_artifacts[key]
                print(f"🧹 Cleaned up {len(old_artifacts)} old artifact timestamps")
                
        except Exception as e:
            print(f"⚠️ Error during resource cleanup: {e}")

    def _finalize_streaming_heatmap(self, chat_id: str, state: dict, choice: dict) -> None:
        """
        Generate the final heatmap artifact when streaming is complete and inject it into the final chunk
        """
        try:
            tokens = state["tokens"]
            token_logprobs = state["logprobs"]
            top_logprobs = state["top_logprobs"]
            
            print(f"[DEBUG] _finalize_streaming_heatmap called with {len(tokens)} tokens")
            
            if len(tokens) == 0:
                print(f"[DEBUG] No tokens to process in finalization")
                return
            
            # Generate final heatmap HTML
            heatmap_html = self._generate_heatmap_html(tokens, token_logprobs, top_logprobs, self.conversation_turn_count, is_streaming=False)
            
            print(f"[DEBUG] Generated heatmap HTML length: {len(heatmap_html) if heatmap_html else 0}")
            
            if heatmap_html:
                # Add final heatmap to the delta content
                if "delta" not in choice:
                    choice["delta"] = {}
                    print(f"[DEBUG] Created new delta in choice")
                
                # Add the heatmap as additional content to the final streaming chunk
                final_update = f"\n\n{heatmap_html}"
                
                if "content" in choice["delta"]:
                    choice["delta"]["content"] += final_update
                    print(f"[DEBUG] Appended heatmap to existing delta content")
                else:
                    choice["delta"]["content"] = final_update
                    print(f"[DEBUG] Set heatmap as new delta content")
                
                # Mark as processed to avoid outlet duplication
                unique_message_key = f"{chat_id}:streaming-final:processed"
                self.processed_messages.add(unique_message_key)
                
                # Record artifact generation timestamp to prevent duplicates
                self.recent_artifacts[f"turn{self.conversation_turn_count}"] = time.time()
                
                print(f"✅ FINAL STREAMING HEATMAP GENERATED and injected - {len(tokens)} tokens analyzed")
            else:
                print(f"[DEBUG] Heatmap HTML generation returned empty result")
            
        except Exception as e:
            print(f"❌ Error finalizing streaming heatmap: {e}")
            import traceback
            traceback.print_exc()

    def _reconstruct_content_from_logprobs(self, logprobs_data) -> str:
        """
        Reconstruct content from logprobs tokens when content is empty.
        This handles the case where streaming is off but content="" and all logprobs are present.
        """
        print(f"🔧 _reconstruct_content_from_logprobs called")
        
        # Handle different logprobs data formats
        if isinstance(logprobs_data, dict) and "content" in logprobs_data:
            logprobs_data = logprobs_data["content"]
        
        if not isinstance(logprobs_data, list):
            print(f"❌ Cannot reconstruct - logprobs data is not a list: {type(logprobs_data)}")
            return ""
        
        tokens = []
        for item in logprobs_data:
            if isinstance(item, dict):
                token = item.get("token", "")
                tokens.append(token)
            else:
                print(f"⚠️ Unexpected logprob item type: {type(item)}")
        
        reconstructed = "".join(tokens)
        print(f"✅ Reconstructed {len(tokens)} tokens into {len(reconstructed)} chars")
        return reconstructed

    def _create_heatmap_html(self, content: str, logprobs_data: list, turn: int) -> str:
        """
        Create HTML with heatmap styling based on OpenAI-compatible logprobs format with turn-based versioning.
        In Open WebUI's outlet, logprobs_data is already extracted and normalized to a consistent format.
        Returns empty string if no valid logprobs data is available.
        """
        print(f"🔍 _create_heatmap_html called with:")
        print(f"   - Content length: {len(content) if content else 0}")
        print(f"   - Logprobs data type: {type(logprobs_data)}")
        print(f"   - Turn: {turn}")
        
        # Early return if no logprobs data - content will be displayed as plain text
        if not logprobs_data:
            print(f"❌ No logprobs data provided - returning empty string")
            print(f"💡 Content will be displayed as plain text without heatmap")
            return ""
        
        if not isinstance(logprobs_data, list):
            print(f"❌ Invalid logprobs data format - expected list, got {type(logprobs_data)}")
            if isinstance(logprobs_data, dict):
                print(f"   - Dict keys: {list(logprobs_data.keys())}")
                # Try to extract from dict format
                if "content" in logprobs_data and isinstance(logprobs_data["content"], list):
                    print(f"   - Found content key with list, using that")
                    logprobs_data = logprobs_data["content"]
                else:
                    print(f"   - No content key or not a list - returning empty string")
                    return ""
            else:
                print(f"❌ Cannot process logprobs data - returning empty string")
                return ""
        
        if not logprobs_data:
            print(f"❌ Empty logprobs data list - returning empty string")
            return ""

        print(f"✅ Processing {len(logprobs_data)} logprob entries")

        tokens = []
        token_logprobs = []
        top_logprobs = []

        for i, item in enumerate(logprobs_data):
            if isinstance(item, dict):
                token = item.get("token", "")
                logprob = item.get("logprob", None)
                top_lp = item.get("top_logprobs", [])
                
                tokens.append(token)
                token_logprobs.append(logprob)
                top_logprobs.append(top_lp)
                
                if i < 3:  # Log first few for debugging
                    print(f"   - Token {i}: '{token}' (logprob: {logprob}, alternatives: {len(top_lp)})")
            else:
                print(f"   - Item {i} is not a dict: {type(item)}")

        print(f"✅ Extracted {len(tokens)} tokens, calling _generate_heatmap_html")
        result = self._generate_heatmap_html(tokens, token_logprobs, top_logprobs, turn)
        print(f"✅ _generate_heatmap_html returned {len(result) if result else 0} characters")
        
        return result

    def _generate_heatmap_html(self, tokens: list, token_logprobs: list, top_logprobs: list, turn: int, is_streaming: bool = False) -> str:
        """
        Generate HTML code block that Open WebUI will render as an artifact with turn-based versioning.
        Supports both streaming updates and final artifacts.
        Returns empty string if no valid logprobs data is available.
        """
        print(f"🔍 _generate_heatmap_html called with:")
        print(f"   - Tokens: {len(tokens)}")
        print(f"   - Token logprobs: {len(token_logprobs)}")
        print(f"   - Top logprobs: {len(top_logprobs)}")
        print(f"   - Turn: {turn}")
        print(f"   - Is streaming: {is_streaming}")

        # Early validation - return empty if no meaningful data
        if not tokens:
            print(f"❌ No tokens to process - returning empty string")
            return ""
        
        if not token_logprobs:
            print(f"❌ No token logprobs to process - returning empty string")
            print(f"💡 This likely means logprobs were truncated due to token limit")
            return ""

        # Validate that we have matching data
        if len(tokens) != len(token_logprobs):
            print(f"❌ Token count mismatch: {len(tokens)} tokens vs {len(token_logprobs)} logprobs")
            # Truncate to minimum length
            min_len = min(len(tokens), len(token_logprobs))
            tokens = tokens[:min_len]
            token_logprobs = token_logprobs[:min_len]
            top_logprobs = top_logprobs[:min_len] if len(top_logprobs) > min_len else top_logprobs

        # Calculate probability ranges for color mapping
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if not valid_logprobs:
            print(f"❌ No valid logprobs found - returning empty string")
            print(f"💡 All logprobs are None, likely due to truncation at token limit")
            return ""
            
        print(f"✅ Found {len(valid_logprobs)} valid logprobs out of {len(token_logprobs)} tokens")
        
        min_logprob = min(valid_logprobs)
        max_logprob = max(valid_logprobs)
        logprob_range = max_logprob - min_logprob if max_logprob != min_logprob else 1

        print(f"✅ Logprob range: {min_logprob:.3f} to {max_logprob:.3f}")

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

        # Add unique artifact ID with different prefixes for streaming vs outlet
        if is_streaming:
            artifact_prefix = "stream"
        else:
            artifact_prefix = "outlet"
        
        # Use microseconds for better uniqueness to prevent ID conflicts
        unique_timestamp = int(time.time() * 1000000)  # microseconds
        artifact_id = f"logprobs-heatmap-{artifact_prefix}-turn{turn}-{unique_timestamp}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine title and status based on streaming state
        title_prefix = "🔄 Live" if is_streaming else "🎯 Final"
        status_text = f"Streaming: {len(tokens)} tokens so far..." if is_streaming else f"Complete: {len(tokens)} tokens analyzed"

        print(f"✅ Generating HTML artifact with ID: {artifact_id}")
        print(f"✅ Status: {status_text}")

        # Create the full HTML artifact with turn-based versioning
        artifact_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_prefix} Token Probability Heatmap - Turn {turn}</title>
    <!-- Conversation Turn: {turn} | Generated: {timestamp} | Streaming: {is_streaming} -->
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
            background: {"#fff3cd" if is_streaming else "#e8f5e8"};
            border-radius: 5px;
            padding: 8px;
            margin: 10px 0;
            font-size: 0.9rem;
            color: {"#856404" if is_streaming else "#2d5a2d"};
            border-left: 4px solid {"#ffc107" if is_streaming else "#4caf50"};
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
            <h1 class="title">{title_prefix} Token Probability Heatmap</h1>
            <p class="subtitle">💡 Hover over tokens to see detailed probability information and alternatives</p>
            <div class="turn-info">
                📦 Conversation Turn: {turn} | Generated: {timestamp} | {status_text}
            </div>
        </div>

        <div class="text-content" id="text-content">{"".join(html_content)}</div>
        
        <div class="footer">
            📊 **Stats:** {status_text} with ranking-based confidence visualization
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
        console.log('{title_prefix} Logprobs Heatmap Artifact Turn {turn} loaded at {timestamp}');
    </script>
</body>
</html>'''

        # Return as a code block that Open WebUI will detect and render as an artifact
        result = f"\n\n{title_prefix} **Token Probability Heatmap Generated!** (Conversation Turn {turn})\n\n```html\n{artifact_html}\n```\n\n📊 **Stats:** {status_text} | Artifact ID: `{artifact_id}`\n"
        
        print(f"✅ HTML artifact generated successfully!")
        print(f"   - Result length: {len(result)} characters")
        print(f"   - Artifact ID: {artifact_id}")
        
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
__version__ = "8.3.0"
__author__ = "Assistant"
__description__ = "Real-time streaming token probability heatmap generator. Creates interactive HTML artifacts that update progressively as tokens are generated during streaming. Features live updates during generation and eliminates duplicate processing. Each conversation turn gets its own unique artifact with streaming and final versions. Works with all LLM providers through Open WebUI's unified OpenAI-compatible format. Intelligently handles logprobs truncation - displays plain text when logprobs are not available, generates heatmaps only when logprobs data is present. Compatible with configurable backend logprobs limits that prevent 'Chunk too big' errors."
