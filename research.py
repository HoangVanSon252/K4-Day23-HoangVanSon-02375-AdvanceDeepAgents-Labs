import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"

def slugify(topic):
    slug = re.sub(r'[^\w\s-]', '', topic).strip().lower()
    slug = re.sub(r'[\s-]+', '-', slug)
    slug = slug[:60]
    return slug if slug else "topic"

def build_prompt(topic):
    return f"Research the following topic and write a report: {topic}"

def summarize(messages, elapsed, model_name):
    subagent_calls = 0
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0
    
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name:
                    tool_calls[name] += 1
                    if name == "task":
                        subagent_calls += 1
                        
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            input_tokens += msg.usage_metadata.get("input_tokens", 0)
            output_tokens += msg.usage_metadata.get("output_tokens", 0)
            
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens
        }
    }

def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_content = files.get(REPORT_PATH)
    sources_content = files.get(SOURCES_PATH)
    
    if not report_content or not sources_content:
        raise RuntimeError("Report or sources.json is missing or empty")
        
    try:
        sources_json = json.loads(sources_content.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"sources.json is invalid JSON: {e}")
        
    if not report_content.strip():
        raise RuntimeError("Report is empty")
        
    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    source_families = sorted(list(set(s.get("source") for s in sources_json if s.get("source"))))
    
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources_json),
        "source_families": source_families
    }
    
    with open(reports_dir / f"{slug}.md", "wb") as f:
        f.write(report_content)
        
    with open(reports_dir / f"{slug}.sources.json", "wb") as f:
        f.write(sources_content)
        
    with open(reports_dir / f"{slug}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return reports_dir / f"{slug}.md"

def main(topic):
    if not topic.strip():
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2
        
    model = make_model()
    model_name = os.environ.get("LAB_MODEL", "unknown_model")
    start = time.monotonic()
    
    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()
        })
        
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000}
            )
        except Exception as e:
            print(f"FAILED during agent execution: {e}", file=sys.stderr)
            # Fall through to save_outputs to see if any report was successfully written anyway
            result = {"messages": []} # Prevent unhandled exception if result was not created
        
        elapsed = time.monotonic() - start
        try:
            report_path = save_outputs(backend, topic, result.get("messages", []), elapsed, model_name)
            print(f"Report saved to {report_path}")
        except RuntimeError as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 1
            
    return 0

if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
