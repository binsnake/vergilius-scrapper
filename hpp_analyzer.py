"""
HPP Analyzer with clangd Integration

Analyzes C++ header files using clangd to detect:
- Undefined/missing symbols
- Ordering errors (incomplete types)
- Type mismatches

Then fetches missing definitions from the Vergilius Project.

Usage:
    python hpp_analyzer.py <input.hpp> <vergilius_base_url> [output.hpp]
    
Example:
    python hpp_analyzer.py my_structs.hpp https://www.vergiliusproject.com/kernels/x64/windows-11/24h2 output.hpp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vergilius_scraper import (
    VergiliusScraper,
    HeaderGenerator,
    StructureInfo,
    PRIMITIVE_TYPES,
)


@dataclass
class ClangDiagnostic:
    """Represents a diagnostic from clangd/clang."""
    file: str
    line: int
    column: int
    severity: str  # error, warning, note
    message: str
    code: Optional[str] = None


@dataclass 
class AnalysisResult:
    """Result of analyzing a header file with clangd."""
    undefined_types: set[str] = field(default_factory=set)
    incomplete_types: set[str] = field(default_factory=set)
    ordering_errors: list[str] = field(default_factory=list)
    all_diagnostics: list[ClangDiagnostic] = field(default_factory=list)
    has_errors: bool = False


class ClangdAnalyzer:
    """
    Analyzes C++ headers using clangd for accurate error detection.
    
    Falls back to clang if clangd is not available.
    """
    
    def __init__(self) -> None:
        self.clangd_path = self._find_clangd()
        self.clang_path = self._find_clang()
    
    def _find_clangd(self) -> Optional[str]:
        """Find clangd executable."""
        path = shutil.which("clangd")
        if path:
            return path
        # Common locations
        for candidate in [
            r"C:\Program Files\LLVM\bin\clangd.exe",
            r"C:\Program Files (x86)\LLVM\bin\clangd.exe",
            "/usr/bin/clangd",
            "/usr/local/bin/clangd",
        ]:
            if Path(candidate).exists():
                return candidate
        return None
    
    def _find_clang(self) -> Optional[str]:
        """Find clang executable (fallback)."""
        path = shutil.which("clang")
        if path:
            return path
        path = shutil.which("clang++")
        if path:
            return path
        for candidate in [
            r"C:\Program Files\LLVM\bin\clang.exe",
            r"C:\Program Files (x86)\LLVM\bin\clang.exe",
            "/usr/bin/clang",
            "/usr/local/bin/clang",
        ]:
            if Path(candidate).exists():
                return candidate
        return None
    
    def _parse_clang_diagnostics(self, output: str) -> list[ClangDiagnostic]:
        """Parse clang's diagnostic output."""
        diagnostics: list[ClangDiagnostic] = []
        
        # Pattern: file:line:col: severity: message
        pattern = r'^(.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+)$'
        
        for line in output.split('\n'):
            match = re.match(pattern, line)
            if match:
                diagnostics.append(ClangDiagnostic(
                    file=match.group(1),
                    line=int(match.group(2)),
                    column=int(match.group(3)),
                    severity=match.group(4),
                    message=match.group(5)
                ))
        
        return diagnostics
    
    def _extract_undefined_types(self, diagnostics: list[ClangDiagnostic]) -> set[str]:
        """Extract undefined type names from diagnostics."""
        undefined: set[str] = set()
        
        patterns = [
            # "unknown type name '_TYPENAME'"
            r"unknown type name '(_\w+)'",
            # "incomplete type 'struct _TYPENAME'"
            r"incomplete type '(?:struct|union|enum)\s+(_\w+)'",
            # "variable has incomplete type 'struct _TYPENAME'"
            r"variable has incomplete type '(?:struct|union|enum)\s+(_\w+)'",
            # "field has incomplete type '_TYPENAME'"
            r"field has incomplete type '(?:struct|union|enum)?\s*(_\w+)'",
            # "member access into incomplete type '_TYPENAME'"
            r"member access into incomplete type '(?:struct|union|enum)?\s*(_\w+)'",
            # forward declaration of 'struct _TYPENAME'
            r"forward declaration of '(?:struct|union|enum)\s+(_\w+)'",
        ]
        
        for diag in diagnostics:
            if diag.severity in ('error', 'warning'):
                for pattern in patterns:
                    matches = re.findall(pattern, diag.message)
                    for m in matches:
                        if m.startswith('_') and m not in PRIMITIVE_TYPES:
                            undefined.add(m)
        
        return undefined
    
    def _extract_ordering_errors(self, diagnostics: list[ClangDiagnostic]) -> list[str]:
        """Extract ordering-related errors."""
        ordering: list[str] = []
        
        ordering_patterns = [
            r"incomplete type",
            r"forward declaration",
            r"used before",
        ]
        
        for diag in diagnostics:
            if diag.severity == 'error':
                for pattern in ordering_patterns:
                    if re.search(pattern, diag.message, re.IGNORECASE):
                        ordering.append(f"{diag.file}:{diag.line}: {diag.message}")
                        break
        
        return ordering
    
    def analyze_with_clang(self, filepath: Path) -> AnalysisResult:
        """Analyze a file using clang (compile check)."""
        result = AnalysisResult()
        
        if not self.clang_path:
            print("[WARN] clang not found, falling back to regex analysis", file=sys.stderr)
            return self._fallback_regex_analysis(filepath)
        
        # Create a temporary source file that includes the header
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
            tmp.write(f'#include "{filepath.absolute()}"\n')
            tmp.write('int main() { return 0; }\n')
            tmp_path = tmp.name
        
        try:
            # Run clang with syntax-only check
            cmd = [
                self.clang_path,
                '-fsyntax-only',
                '-std=c++17',
                '-target', 'x86_64-pc-windows-msvc',  # Windows x64 target
                '-fms-extensions',
                '-fms-compatibility',
                '-Wno-pragma-pack',
                tmp_path
            ]
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse diagnostics from stderr
            diagnostics = self._parse_clang_diagnostics(proc.stderr)
            result.all_diagnostics = diagnostics
            result.has_errors = proc.returncode != 0
            
            # Extract specific error types
            result.undefined_types = self._extract_undefined_types(diagnostics)
            result.ordering_errors = self._extract_ordering_errors(diagnostics)
            
            # Incomplete types are a subset of undefined
            for diag in diagnostics:
                if 'incomplete type' in diag.message:
                    matches = re.findall(r"'(?:struct|union|enum)?\s*(_\w+)'", diag.message)
                    for m in matches:
                        if m.startswith('_'):
                            result.incomplete_types.add(m)
            
        except subprocess.TimeoutExpired:
            print("[ERROR] clang timed out", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] clang failed: {e}", file=sys.stderr)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        return result
    
    def _fallback_regex_analysis(self, filepath: Path) -> AnalysisResult:
        """Fallback to regex-based analysis if clang is not available."""
        result = AnalysisResult()
        content = filepath.read_text(encoding='utf-8')
        
        # Extract defined types
        defined: set[str] = set()
        definition_pattern = r'(?:struct|union|enum)\s+(_\w+)\s*(?:\{|;)'
        for match in re.finditer(definition_pattern, content):
            defined.add(match.group(1))
        
        # Extract referenced types
        referenced: set[str] = set()
        usage_pattern = r'(?:struct|union|enum)\s+(_\w+)'
        for match in re.finditer(usage_pattern, content):
            referenced.add(match.group(1))
        
        # Missing = referenced but not defined
        result.undefined_types = {
            t for t in referenced 
            if t not in defined and t not in PRIMITIVE_TYPES
        }
        
        return result
    
    def analyze(self, filepath: Path) -> AnalysisResult:
        """Analyze a header file for errors."""
        return self.analyze_with_clang(filepath)


class MissingSymbolFetcher:
    """Fetches missing symbols from the Vergilius Project using async HTTP."""
    
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.scraper = VergiliusScraper()
        self.scraper._url_prefix = self.base_url + '/'
    
    async def fetch_missing_async(
        self,
        missing_types: set[str],
        existing_types: set[str],
        max_depth: int = 50
    ) -> dict[str, StructureInfo]:
        """Fetch missing type definitions asynchronously."""
        # Mark existing types
        for t in existing_types:
            self.scraper.structures[t] = StructureInfo(
                name=t,
                size_bytes=0,
                raw_definition="// Already defined",
                url="",
                dependencies=set()
            )
        
        # Build initial URLs
        initial_urls = [f"{self.base_url}/{t}" for t in missing_types if t not in PRIMITIVE_TYPES]
        
        if not initial_urls:
            return {}
        
        # Use the async scraper
        await self.scraper.scrape_recursive_async(initial_urls[0], max_depth)
        
        # Scrape remaining initial types that weren't dependencies
        for url in initial_urls[1:]:
            type_name = url.split('/')[-1]
            if type_name not in self.scraper.structures:
                self.scraper.visited_urls.discard(url)  # Allow re-visit
                await self.scraper.scrape_recursive_async(url, max_depth)
        
        # Return only newly fetched structures
        fetched = {
            name: info for name, info in self.scraper.structures.items()
            if name not in existing_types and info.raw_definition != "// Already defined"
        }
        
        return fetched
    
    def fetch_missing(
        self,
        missing_types: set[str],
        existing_types: set[str],
        max_depth: int = 50
    ) -> dict[str, StructureInfo]:
        """Synchronous wrapper for async fetching."""
        return asyncio.run(self.fetch_missing_async(missing_types, existing_types, max_depth))


def find_first_struct_usage(content: str, type_name: str) -> int:
    """
    Find the line number where a type is first used (not defined).
    
    Returns -1 if not found.
    """
    lines = content.split('\n')
    
    # Pattern to find usage of the type (not its definition)
    # Look for: struct _NAME field, struct _NAME*, or just _NAME as a type
    usage_patterns = [
        rf'\b(?:struct|union|enum)\s+{re.escape(type_name)}\s+\w',  # struct _NAME field
        rf'\b(?:struct|union|enum)\s+{re.escape(type_name)}\s*\*',  # struct _NAME*
        rf'\b{re.escape(type_name)}\s+\w',  # _NAME field (without struct keyword)
    ]
    
    # Pattern for definition (to exclude)
    def_pattern = rf'^(?://.*\n)*(?:struct|union|enum)\s+{re.escape(type_name)}\s*\{{'
    
    for i, line in enumerate(lines):
        # Skip if this is the definition
        if re.search(def_pattern, line):
            continue
        
        for pattern in usage_patterns:
            if re.search(pattern, line):
                return i
    
    return -1


def merge_hpp_content(
    original_content: str,
    new_structures: dict[str, StructureInfo],
    existing_types: set[str]
) -> str:
    """
    Merge new structure definitions into existing HPP content.
    
    Inserts new structures BEFORE the first structure that uses them,
    to ensure proper compilation order.
    """
    if not new_structures:
        return original_content
    
    generator = HeaderGenerator(new_structures)
    sorted_names = generator._topological_sort()
    
    # Build the new definitions block
    new_lines: list[str] = []
    new_lines.append("")
    new_lines.append("// ============================================")
    new_lines.append("// Additional structures fetched from Vergilius")
    new_lines.append("// ============================================")
    new_lines.append("")
    
    # Forward declarations
    new_lines.append("// Forward declarations for new types")
    for name in sorted_names:
        info = new_structures[name]
        keyword = generator._extract_struct_keyword(info.raw_definition)
        new_lines.append(f"{keyword} {name};")
    new_lines.append("")
    
    # Definitions
    for name in sorted_names:
        info = new_structures[name]
        cleaned_def = generator._clean_definition(info.raw_definition)
        new_lines.append(cleaned_def)
        new_lines.append("")
        
        if info.size_bytes > 0:
            new_lines.append(f"static_assert(sizeof({name}) == 0x{info.size_bytes:x}, \"Size mismatch for {name}\");")
            new_lines.append("")
    
    lines = original_content.rstrip().split('\n')
    
    # Find the earliest usage of any new structure
    earliest_usage = len(lines)
    for name in sorted_names:
        usage_line = find_first_struct_usage(original_content, name)
        if usage_line != -1 and usage_line < earliest_usage:
            earliest_usage = usage_line
    
    # If we found a usage, insert before it
    # Otherwise, insert before the final #endif
    if earliest_usage < len(lines):
        # Find the start of the struct/block that contains this usage
        # Go back to find the line that starts the struct definition
        insert_pos = earliest_usage
        for i in range(earliest_usage - 1, -1, -1):
            line = lines[i].strip()
            # Look for start of struct/union/enum definition
            if re.match(r'^(?://.*)?(?:struct|union|enum)\s+_\w+\s*$', line):
                insert_pos = i
                break
            if re.match(r'^(?://.*)?(?:struct|union|enum)\s+_\w+\s*\{', line):
                insert_pos = i
                break
            # Check for size comment that precedes struct
            if re.match(r'^//0x[0-9a-fA-F]+\s+bytes', line):
                insert_pos = i
                break
            # Stop if we hit another complete definition or a blank line after a }
            if line == '' and i > 0 and lines[i-1].strip().endswith('};'):
                insert_pos = i + 1
                break
    else:
        # Fallback: insert before final #endif
        insert_pos = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('#endif'):
                insert_pos = i
                break
    
    result_lines = lines[:insert_pos] + new_lines + lines[insert_pos:]
    return '\n'.join(result_lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze HPP file with clangd and fetch missing symbols from Vergilius",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python hpp_analyzer.py my_structs.hpp https://www.vergiliusproject.com/kernels/x64/windows-11/24h2 output.hpp
    
    # Just analyze without fetching:
    python hpp_analyzer.py my_structs.hpp --analyze-only
        """
    )
    parser.add_argument("input", help="Input HPP file to analyze")
    parser.add_argument("base_url", nargs="?", help="Vergilius base URL")
    parser.add_argument("output", nargs="?", help="Output HPP file")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't fetch")
    parser.add_argument("--max-depth", type=int, default=50, help="Maximum recursion depth")
    parser.add_argument("--in-place", action="store_true", help="Modify input file in place")
    parser.add_argument("--no-clang", action="store_true", help="Skip clang analysis, use regex only")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    print(f"[INFO] Analyzing: {args.input}")
    
    # Analyze with clangd/clang
    analyzer = ClangdAnalyzer()
    
    if args.no_clang:
        result = analyzer._fallback_regex_analysis(input_path)
    else:
        result = analyzer.analyze(input_path)
    
    # Report findings
    print(f"[INFO] Analysis complete:")
    print(f"  - Errors found: {result.has_errors}")
    print(f"  - Undefined types: {len(result.undefined_types)}")
    print(f"  - Incomplete types: {len(result.incomplete_types)}")
    print(f"  - Ordering errors: {len(result.ordering_errors)}")
    
    if result.undefined_types:
        print(f"\n[INFO] Undefined types:")
        for t in sorted(result.undefined_types):
            print(f"  - {t}")
    
    if result.ordering_errors:
        print(f"\n[INFO] Ordering errors:")
        for err in result.ordering_errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(result.ordering_errors) > 10:
            print(f"  ... and {len(result.ordering_errors) - 10} more")
    
    if args.analyze_only:
        return 0 if not result.has_errors else 1
    
    # Combine undefined and incomplete types for fetching
    missing = result.undefined_types | result.incomplete_types
    
    if not missing:
        print("[INFO] No missing types to fetch!")
        return 0
    
    if not args.base_url:
        print("[ERROR] base_url is required to fetch missing types", file=sys.stderr)
        return 1
    
    # Get existing defined types from file (regex-based for speed)
    content = input_path.read_text(encoding='utf-8')
    existing_types: set[str] = set()
    for match in re.finditer(r'(?:struct|union|enum)\s+(_\w+)\s*[{;]', content):
        existing_types.add(match.group(1))
    
    # Fetch missing types
    print(f"\n[INFO] Fetching {len(missing)} missing types from Vergilius...")
    
    fetcher = MissingSymbolFetcher(args.base_url)
    new_structures = fetcher.fetch_missing(missing, existing_types, max_depth=args.max_depth)
    
    if not new_structures:
        print("[WARN] Could not fetch any missing types")
        return 1
    
    print(f"\n[INFO] Fetched {len(new_structures)} new structures")
    
    # Merge and write
    merged_content = merge_hpp_content(content, new_structures, existing_types)
    
    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_complete")
    
    output_path.write_text(merged_content, encoding='utf-8')
    print(f"[INFO] Written to: {output_path}")
    
    # Verify with clang again
    if not args.no_clang:
        print("\n[INFO] Verifying output with clang...")
        verify_result = analyzer.analyze(output_path)
        
        if verify_result.has_errors:
            still_missing = verify_result.undefined_types | verify_result.incomplete_types
            if still_missing:
                print(f"[WARN] Still missing {len(still_missing)} types after fetch:")
                for t in sorted(still_missing)[:10]:
                    print(f"  - {t}")
        else:
            print("[INFO] Output verified successfully - no errors!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
