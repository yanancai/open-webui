import json
from uuid import uuid4
from open_webui.utils.misc import (
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)

# Configuration: Limit logprobs to prevent aiohttp chunk size issues
# Based on analysis: 500 tokens with logprobs = ~439KB, well within 512KB limit
# This preserves the most useful logprobs data while preventing "Chunk too big" errors
LOGPROBS_TOKEN_LIMIT = 500


def convert_ollama_tool_call_to_openai(tool_calls: list) -> list:
    openai_tool_calls = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        openai_tool_call = {
            "index": tool_call.get("index", function.get("index", 0)),
            "id": tool_call.get("id", f"call_{str(uuid4())}"),
            "type": "function",
            "function": {
                "name": function.get("name", ""),
                "arguments": json.dumps(function.get("arguments", {})),
            },
        }
        openai_tool_calls.append(openai_tool_call)
    return openai_tool_calls


def convert_ollama_usage_to_openai(data: dict) -> dict:
    return {
        "response_token/s": (
            round(
                (
                    (
                        data.get("eval_count", 0)
                        / ((data.get("eval_duration", 0) / 10_000_000))
                    )
                    * 100
                ),
                2,
            )
            if data.get("eval_duration", 0) > 0
            else "N/A"
        ),
        "prompt_token/s": (
            round(
                (
                    (
                        data.get("prompt_eval_count", 0)
                        / ((data.get("prompt_eval_duration", 0) / 10_000_000))
                    )
                    * 100
                ),
                2,
            )
            if data.get("prompt_eval_duration", 0) > 0
            else "N/A"
        ),
        "total_duration": data.get("total_duration", 0),
        "load_duration": data.get("load_duration", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_tokens": int(
            data.get("prompt_eval_count", 0)
        ),  # This is the OpenAI compatible key
        "prompt_eval_duration": data.get("prompt_eval_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "completion_tokens": int(
            data.get("eval_count", 0)
        ),  # This is the OpenAI compatible key
        "eval_duration": data.get("eval_duration", 0),
        "approximate_total": (lambda s: f"{s // 3600}h{(s % 3600) // 60}m{s % 60}s")(
            (data.get("total_duration", 0) or 0) // 1_000_000_000
        ),
        "total_tokens": int(  # This is the OpenAI compatible key
            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        ),
        "completion_tokens_details": {  # This is the OpenAI compatible key
            "reasoning_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def convert_response_ollama_to_openai(ollama_response: dict) -> dict:
    model = ollama_response.get("model", "ollama")
    message_content = ollama_response.get("message", {}).get("content", "")
    reasoning_content = ollama_response.get("message", {}).get("thinking", None)
    tool_calls = ollama_response.get("message", {}).get("tool_calls", None)
    logprobs = ollama_response.get("message", {}).get("logprobs", None)
    
    openai_tool_calls = None

    if tool_calls:
        openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)

    data = ollama_response

    usage = convert_ollama_usage_to_openai(data)

    response = openai_chat_completion_message_template(
        model, message_content, reasoning_content, openai_tool_calls, usage, logprobs
    )
    return response


async def convert_streaming_response_ollama_to_openai(ollama_streaming_response):
    chunk_count = 0
    total_content_size = 0
    logprobs_token_count = 0
    
    try:
        print(f"[DEBUG] Starting ollama streaming conversion with logprobs limit: {LOGPROBS_TOKEN_LIMIT} tokens")
        async for data in ollama_streaming_response.body_iterator:
            chunk_count += 1
            total_content_size += len(data) if isinstance(data, (bytes, str)) else 0
            
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON decode error on chunk {chunk_count}: {e}")
                continue
            
            # Debug: Print chunk info when logprobs are present
            has_logprobs = data.get("message", {}).get("logprobs") is not None
            if has_logprobs:
                print(f"[DEBUG] Chunk {chunk_count} has logprobs, size: {total_content_size} bytes")

            model = data.get("model", "ollama")
            message_content = data.get("message", {}).get("content", None)
            reasoning_content = data.get("message", {}).get("thinking", None)
            tool_calls = data.get("message", {}).get("tool_calls", None)
            logprobs = data.get("message", {}).get("logprobs", None)
            openai_tool_calls = None

            # Handle logprobs truncation to prevent chunk size issues
            if logprobs:
                if logprobs_token_count < LOGPROBS_TOKEN_LIMIT:
                    # Count tokens in this chunk's logprobs
                    if isinstance(logprobs, dict) and "content" in logprobs:
                        tokens_in_chunk = len(logprobs.get("content", []))
                    elif isinstance(logprobs, list):
                        tokens_in_chunk = len(logprobs)
                    else:
                        tokens_in_chunk = 1  # Single token logprob
                    
                    # Check if adding this chunk would exceed the limit
                    if logprobs_token_count + tokens_in_chunk > LOGPROBS_TOKEN_LIMIT:
                        # Truncate to stay within limit
                        remaining_tokens = LOGPROBS_TOKEN_LIMIT - logprobs_token_count
                        if remaining_tokens > 0:
                            if isinstance(logprobs, dict) and "content" in logprobs:
                                logprobs["content"] = logprobs["content"][:remaining_tokens]
                            elif isinstance(logprobs, list):
                                logprobs = logprobs[:remaining_tokens]
                            
                            logprobs_token_count = LOGPROBS_TOKEN_LIMIT
                            print(f"[DEBUG] Logprobs truncated at {LOGPROBS_TOKEN_LIMIT} tokens (chunk {chunk_count})")
                        else:
                            # Already at limit, drop logprobs entirely
                            logprobs = None
                            print(f"[DEBUG] Logprobs dropped after {LOGPROBS_TOKEN_LIMIT} tokens (chunk {chunk_count})")
                    else:
                        # Within limit, keep all logprobs
                        logprobs_token_count += tokens_in_chunk
                        print(f"[DEBUG] Logprobs kept: {logprobs_token_count}/{LOGPROBS_TOKEN_LIMIT} tokens (chunk {chunk_count})")
                else:
                    # Already exceeded limit, drop logprobs
                    logprobs = None
                    if chunk_count % 10 == 0:  # Log every 10th chunk to avoid spam
                        print(f"[DEBUG] Logprobs limit exceeded, dropping logprobs (chunk {chunk_count})")

            if logprobs:
                print(f"[DEBUG] Found logprobs in chunk {chunk_count}: {type(logprobs)}")

            if tool_calls:
                openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)

            done = data.get("done", False)

            usage = None
            if done:
                usage = convert_ollama_usage_to_openai(data)
                print(f"[DEBUG] Final chunk {chunk_count}, total size: {total_content_size} bytes")

            data = openai_chat_chunk_message_template(
                model, message_content, reasoning_content, openai_tool_calls, usage, logprobs
            )

            line = f"data: {json.dumps(data)}\n\n"
            yield line

        yield "data: [DONE]\n\n"
        print(f"[DEBUG] Streaming conversion completed successfully: {chunk_count} chunks, {total_content_size} bytes")
        if logprobs_token_count > 0:
            print(f"[DEBUG] Logprobs processed for {logprobs_token_count} tokens (limit: {LOGPROBS_TOKEN_LIMIT})")
    except Exception as e:
        error_msg = str(e).lower()
        if "chunk too big" in error_msg or "line too long" in error_msg:
            print(f"[DEBUG] Chunk size error detected: {e}")
            print(f"[DEBUG] This occurs when logprobs make individual chunks too large for aiohttp")
            print(f"[DEBUG] Processed {chunk_count} chunks, {total_content_size} bytes before error")
            print(f"[DEBUG] Note: Logprobs are now limited to {LOGPROBS_TOKEN_LIMIT} tokens to prevent this issue")
            # Re-raise with more context
            raise Exception(f"Response chunks too large for aiohttp (likely due to logprobs). "
                          f"Processed {chunk_count} chunks, {total_content_size} bytes. "
                          f"Logprobs limit: {LOGPROBS_TOKEN_LIMIT} tokens. "
                          f"Original error: {e}")
        else:
            print(f"[DEBUG] Error in convert_streaming_response_ollama_to_openai after {chunk_count} chunks: {e}")
            print(f"[DEBUG] Total content processed: {total_content_size} bytes")
            if logprobs_token_count > 0:
                print(f"[DEBUG] Logprobs processed: {logprobs_token_count}/{LOGPROBS_TOKEN_LIMIT} tokens")
            raise


async def convert_streaming_ollama_to_complete_response(ollama_streaming_response):
    """
    Convert streaming Ollama response to a complete non-streaming OpenAI response.
    Used when logprobs are requested but user wants non-streaming response.
    """
    model = None
    complete_content = ""
    complete_logprobs = []
    usage_data = None
    tool_calls = None
    reasoning_content = None
    logprobs_token_count = 0
    
    try:
        async for data in ollama_streaming_response.body_iterator:
            data = json.loads(data)
            print(f"[DEBUG] Processing streaming chunk for complete response: {json.dumps(data, indent=2, default=str)}")
            
            model = data.get("model", "ollama")
            message = data.get("message", {})
            content = message.get("content", "")
            logprobs = message.get("logprobs", None)
            
            # Accumulate content
            if content:
                complete_content += content
                
            # Accumulate logprobs with truncation
            if logprobs and logprobs_token_count < LOGPROBS_TOKEN_LIMIT:
                if isinstance(logprobs, list):
                    tokens_to_add = min(len(logprobs), LOGPROBS_TOKEN_LIMIT - logprobs_token_count)
                    complete_logprobs.extend(logprobs[:tokens_to_add])
                    logprobs_token_count += tokens_to_add
                    
                    if logprobs_token_count >= LOGPROBS_TOKEN_LIMIT:
                        print(f"[DEBUG] Logprobs truncated at {LOGPROBS_TOKEN_LIMIT} tokens in complete response")
                else:
                    if logprobs_token_count < LOGPROBS_TOKEN_LIMIT:
                        complete_logprobs.append(logprobs)
                        logprobs_token_count += 1
                        
            # Get tool calls and reasoning from any chunk that has them
            if message.get("tool_calls"):
                tool_calls = message.get("tool_calls")
            if message.get("thinking"):
                reasoning_content = message.get("thinking")
                
            # Get usage data from final chunk
            done = data.get("done", False)
            if done:
                usage_data = convert_ollama_usage_to_openai(data)
                break
        
        print(f"[DEBUG] Complete response - content length: {len(complete_content)}, logprobs count: {len(complete_logprobs)} (limit: {LOGPROBS_TOKEN_LIMIT})")
        
        # Convert to OpenAI format
        openai_tool_calls = None
        if tool_calls:
            openai_tool_calls = convert_ollama_tool_call_to_openai(tool_calls)
        
        response = openai_chat_completion_message_template(
            model, complete_content, reasoning_content, openai_tool_calls, usage_data, complete_logprobs
        )
        return response
    except Exception as e:
        print(f"[DEBUG] Error in convert_streaming_ollama_to_complete_response: {e}")
        raise


def convert_embedding_response_ollama_to_openai(response) -> dict:
    """
    Convert the response from Ollama embeddings endpoint to the OpenAI-compatible format.

    Args:
        response (dict): The response from the Ollama API,
            e.g. {"embedding": [...], "model": "..."}
            or {"embeddings": [{"embedding": [...], "index": 0}, ...], "model": "..."}

    Returns:
        dict: Response adapted to OpenAI's embeddings API format.
            e.g. {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [...], "index": 0},
                    ...
                ],
                "model": "...",
            }
    """
    # Ollama batch-style output
    if isinstance(response, dict) and "embeddings" in response:
        openai_data = []
        for i, emb in enumerate(response["embeddings"]):
            openai_data.append(
                {
                    "object": "embedding",
                    "embedding": emb.get("embedding"),
                    "index": emb.get("index", i),
                }
            )
        return {
            "object": "list",
            "data": openai_data,
            "model": response.get("model"),
        }
    # Ollama single output
    elif isinstance(response, dict) and "embedding" in response:
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": response["embedding"],
                    "index": 0,
                }
            ],
            "model": response.get("model"),
        }
    # Already OpenAI-compatible?
    elif (
        isinstance(response, dict)
        and "data" in response
        and isinstance(response["data"], list)
    ):
        return response

    # Fallback: return as is if unrecognized
    return response
