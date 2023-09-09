"""
OpenAI Client Wrapper.

This class wraps the OpenAI API and provides a few convenience methods.

Caching is useful for speeding up development and saving money :)

- Caching Completions
- Session management
- Retries and backoff
- Rate limiting
- Response postprocessing
"""
from typing import Dict, List, Union

import diskcache
import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

INSERT_API_TOKEN = "[insert]"


def postprocess_completion_response(response: Dict) -> Dict:
    """Postprocess OAI completion API response.

    Standardize the response format and add additional fields.
    Lets us add new LLMs without changing the API.
    """
    return {
        "response": response,
        "num_tokens": response["usage"]["total_tokens"],
        "all_completions": [a["text"] for a in response["choices"]],
        "completion": response["choices"][0]["text"],
        "latency": response["latency"],
        "usage": response["usage"],
    }


class OAIClient:
    def __init__(
        self,
        api_key: str,
        organization_id: Union[str, None] = None,
        cache: Union[diskcache.Cache, None] = None,
    ):
        self._disk_cache = cache
        openai.organization = organization_id
        openai.api_key = api_key

    def _get_cache_key(self, params: dict) -> str:
        """Get cache key for given parameters.

        Args:
            params (dict): Keyword arguments to pass to `openai.Completion.create()`.

        Returns:
            str. Cache key.
        """
        return f"completion:" + ":".join(
            [f"{k}={v}" for k, v in sorted(params.items())]
        )

    def _completion_api_call(self, params: dict) -> Dict:
        """Wrapper so we can time the API call w/o cache."""
        start = time.time()
        response: Dict = openai.Completion.create(**params)  # type:ignore
        response["latency"] = round(time.time() - start, 3)
        return response

    def _complete_with_cache(
        self, params: dict, request_tag: Union[str, None] = None
    ) -> Dict:
        """Call Completion API with caching.

        Args:
            params (dict): See `openai.Completion` documentation.
            request_tag (Union[str, None], optional): Tag for easier request debugging.

        Returns:
            Dict: OAI Completion API response.
        """
        cache_key = self._get_cache_key(params)

        if self._disk_cache is not None:
            cached_response: Dict = self._disk_cache.get(cache_key)  # type: ignore
            if cached_response is not None:
                return cached_response

        response = self._completion_api_call(params)

        if self._disk_cache is not None:
            self._disk_cache.set(cache_key, response, tag=request_tag)

        return response

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.Timeout,
                openai.error.TryAgain,
            )
        ),
    )
    def complete(
        self,
        prompt: str,
        stop: Union[List[str], None] = None,
        n: int = 1,
        best_of: int = 1,
        top_p: int = 1,
        temperature: float = 0,
        logprobs: Union[int, None] = None,
        max_tokens: int = 256,
        frequency_penalty: int = 0,
        presence_penalty: int = 0,
        model: str = "text-davinci-002",
        logit_bias: Union[Dict[str, float], None] = None,
        request_tag: Union[str, None] = None,
        mode: str = "complete",  # or insert
    ) -> Dict:
        """Call OpenAI Completion API.

        TODO(bfortuner): Add Streaming

        See https://beta.openai.com/docs/api-reference/completions for param descriptions.

        Args:
            prompt (str): Prompt to complete.
            request_tag (str): Request Tag to use for cache lookup.
            suffix (str): Appended to prompt for INSERT requests. Example:
                complete(
                    model="text-davinci-002",
                    prompt="We're writing to ",
                    suffix=" this into a paragraph.",
                )
        Returns:
            Dict. Response from OpenAI Completion API.
        """
        suffix = None
        if mode == "insert":
            if prompt.lower().count(INSERT_API_TOKEN) != 1:
                raise ValueError(
                    f"Prompt must contain exactly 1 instance of '{INSERT_API_TOKEN}' token."
                )
            prompt, suffix = re.split(r"\[insert\]", prompt, flags=re.IGNORECASE)

        params = dict(
            prompt=prompt,
            model=model,
            n=n,
            top_p=top_p,
            best_of=best_of,  # we always return all answers so best_of = n
            temperature=temperature,
            logprobs=logprobs,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logit_bias=logit_bias or {},
            suffix=suffix,
        )

        response = self._complete_with_cache(params, request_tag)

        result = postprocess_completion_response(response)
        result["request_params"] = params
        result["request_tag"] = request_tag

        return result
