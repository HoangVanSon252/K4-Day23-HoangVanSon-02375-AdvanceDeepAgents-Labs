import json
import sys
import re

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    problems = []
    if not sources:
        return ["no sources in sources.json"]
    
    seen_urls = set()
    for source in sources:
        n = source.get("n")
        if not isinstance(n, int):
            problems.append(f"n must be an int in source: {source}")
        url = source.get("url", "")
        if not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"url must start with http(s):// in source: {source}")
        if url in seen_urls:
            problems.append(f"duplicate url found: {url}")
        seen_urls.add(url)
        
    parts = report_text.split("## References")
    if len(parts) != 2:
        problems.append("missing '## References' heading or multiple such headings")
        return problems
    
    body = parts[0]
    references_text = parts[1]
    
    cited = set()
    for match in re.finditer(r'\[(\d+)\]', body):
        cited.add(int(match.group(1)))
        
    source_nums = {s.get("n") for s in sources if isinstance(s.get("n"), int)}
    
    for n in cited:
        if n not in source_nums:
            problems.append(f"[{n}] cited but missing from sources.json")
            
    for n in source_nums:
        if n not in cited:
            problems.append(f"source [{n}] never cited")
            
    ref_lines = [line.strip() for line in references_text.split("\n") if re.match(r'^\[\d+\]', line.strip())]
    
    ref_nums = []
    for line in ref_lines:
        match = re.match(r'^\[(\d+)\]', line)
        if match:
            ref_nums.append(int(match.group(1)))
            
    for n in source_nums:
        if ref_nums.count(n) == 0:
            problems.append(f"missing reference line for source {n}")
        elif ref_nums.count(n) > 1:
            problems.append(f"multiple reference lines for source {n}")
            
    for n in ref_nums:
        if n not in source_nums:
            problems.append(f"reference line for [{n}] that is not a source")
            
    for line in ref_lines:
        match = re.match(r'^\[(\d+)\]', line)
        if not match:
            continue
        n = int(match.group(1))
        
        urls = re.findall(r'https?://[^\s\)]+', line)
        if len(urls) != 1:
            problems.append(f"reference line for [{n}] must contain exactly ONE http(s) URL")
        else:
            url = urls[0]
            source_url = next((s.get("url") for s in sources if s.get("n") == n), None)
            if source_url and url != source_url:
                problems.append(f"URL in reference line for [{n}] does not match sources.json")
                
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
