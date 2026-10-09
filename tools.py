import json
import os
import time
import xml.etree.ElementTree as ET
import random
import re

import httpx
from langchain_core.tools import tool

ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            delay = float(e.retry_after) if e.retry_after is not None else base * (2 ** attempt) + random.uniform(0, 0.5)
            delay = min(delay, cap)
            time.sleep(delay)

def safe_request(*args, **kwargs):
    try:
        r = httpx.request(*args, **kwargs)
        if r.status_code in (429, 500, 502, 503, 504):
            retry_after = r.headers.get("Retry-After")
            retry_after = float(retry_after) if retry_after else None
            raise RetryableError(f"HTTP {r.status_code}", retry_after)
        r.raise_for_status()
        return r
    except httpx.TransportError as e:
        raise RetryableError(f"TransportError: {e}")

_last_arxiv_call = 0.0

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_call
    try:
        terms = re.findall(r'\w+', query)
        if not terms:
            return "NO RESULTS"
        
        def _call():
            global _last_arxiv_call
            now = time.monotonic()
            if now - _last_arxiv_call < 3.0:
                time.sleep(3.0 - (now - _last_arxiv_call))
            _last_arxiv_call = time.monotonic()
            
            search_query = " AND ".join(f"all:{t}" for t in terms)
            return safe_request("GET", ARXIV_URL, params={
                "search_query": search_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max(1, min(max_results, 30))
            }, timeout=10.0)

        r = with_retry(_call)
        
        root = ET.fromstring(r.text)
        records = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            id_elem = entry.find('atom:id', ns)
            if id_elem is None: continue
            raw_id = id_elem.text.split('/abs/')[-1]
            paper_id = raw_id.split('v')[0]
            
            published = entry.find('atom:published', ns)
            published_text = published.text[:10] if published is not None else ""
            
            title = entry.find('atom:title', ns)
            title_text = re.sub(r'\s+', ' ', title.text).strip() if title is not None else ""
            
            summary = entry.find('atom:summary', ns)
            summary_text = re.sub(r'\s+', ' ', summary.text).strip()[:600] if summary is not None else ""
            
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": published_text,
                "title": title_text,
                "summary": summary_text
            })
            
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        def _call():
            params = {"limit": max(1, min(limit, 100))}
            if date:
                params["date"] = date
            return safe_request("GET", HF_DAILY_URL, params=params, timeout=10.0)
            
        r = with_retry(_call)
        data = r.json()
        records = []
        kw = keyword.lower()
        for item in data:
            paper = item.get("paper")
            if not paper or not paper.get("id"):
                continue
            id = paper["id"]
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            
            if kw and kw not in title.lower() and kw not in summary.lower():
                continue
                
            records.append({
                "id": id,
                "url": f"https://huggingface.co/papers/{id}",
                "published": paper.get("publishedAt", "")[:10],
                "title": title,
                "summary": summary[:600],
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
            
        records.sort(key=lambda x: x["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        def _call():
            params = {"q": query, "limit": max(1, min(limit, 50))}
            return safe_request("GET", HF_SEARCH_URL, params=params, timeout=10.0)
            
        r = with_retry(_call)
        data = r.json()
        records = []
        for item in data:
            paper = item.get("paper")
            if not paper or not paper.get("id"):
                continue
            id = paper["id"]
            title = paper.get("title", "")
            summary = paper.get("ai_summary") or paper.get("summary", "")
            
            records.append({
                "id": id,
                "url": f"https://huggingface.co/papers/{id}",
                "published": paper.get("publishedAt", "")[:10],
                "title": title,
                "summary": summary[:600],
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def exa_call(method: str, params: dict):
    def _call():
        exa_key = os.environ.get("EXA_API_KEY", "")
        url = EXA_URL
        if exa_key:
            url += f"?exaApiKey={exa_key}"
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=20.0)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RetryableError(f"HTTP {r.status_code}", float(r.headers.get("Retry-After", 0)) or None)
            r.raise_for_status()
            
            lines = r.text.splitlines()
            data_line = next((l for l in lines if l.startswith("data:")), None)
            if not data_line:
                raise Exception("No data line in SSE response")
            
            body = json.loads(data_line[5:])
            if "error" in body:
                raise Exception(f"RPC error: {body['error']}")
                
            result = body.get("result", {})
            
            meta = result.get("_meta", {})
            text_content = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
            
            if meta.get("rateLimitExceeded", False) or ("rate limit" in text_content.lower() and len(text_content) < 500):
                raise RetryableError("Exa rate limit exceeded")
                
            return result
        except httpx.TransportError as e:
            raise RetryableError(f"TransportError: {e}")
            
    try:
        return with_retry(_call, attempts=15, cap=60.0)
    except Exception as e:
        err_msg = str(e)
        exa_key = os.environ.get("EXA_API_KEY", "")
        if exa_key:
            err_msg = err_msg.replace(exa_key, "***")
        raise Exception(err_msg)

@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective if objective else f"find information about {query}"
        params = {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "objective": obj,
                "numResults": num_results
            }
        }
        res = exa_call("tools/call", params)
        content = res.get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        if not text:
            return "NO RESULTS"
        return text
    except Exception as e:
        return f"ERROR: {e}"

@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        params = {
            "name": "web_fetch_exa",
            "arguments": {
                "urls": [url]
            }
        }
        res = exa_call("tools/call", params)
        content = res.get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        if not text:
            return "NO RESULTS"
        return text[:12000]
    except Exception as e:
        return f"ERROR: {e}"

SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
