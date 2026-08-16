//! Turns raw `llvm-cov export` output into one entry per function.
//!
//! LLVM does not count functions the way a person would. A generic function is
//! reported once for every set of types it was used with, and every closure is
//! reported as though it were its own function. So a single function in the
//! source can show up in the raw data many times over.
//!
//! This module adds those duplicates back together, so one entry coming out of
//! here means one function as it is actually written in the source.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustc_demangle::demangle;
use serde::{Deserialize, Serialize};

// llvm-cov export --format=text JSON structures
#[derive(Deserialize)]
struct Export {
    data: Vec<ExportData>,
}

#[derive(Deserialize)]
struct ExportData {
    functions: Vec<Function>,
}

#[derive(Deserialize)]
struct Function {
    name: String,
    filenames: Vec<String>,
    // regions: [line_start, col_start, line_end, col_end, count, ...]
    regions: Vec<Vec<serde_json::Value>>,
}

/// One function's coverage, after merging.
#[derive(Serialize, Deserialize)]
pub struct FunctionReport {
    /// Where this function sits in `CoverageData::functions`.
    ///
    /// Source lines are written out into separate JSON files and looked up by
    /// this number, so it has to stay put once it has been handed out.
    pub id: usize,
    pub demangled: String,
    pub filename: String,
    pub line_start: usize,
    /// How many times each line ran, counting from `line_start`.
    ///
    /// `None` means LLVM does not track that line at all, a blank line or a
    /// closing brace for example, so it counts as neither covered nor missed.
    pub line_counts: Vec<Option<u64>>,
    /// Worked out after merging, so it reflects the summed counts rather than
    /// any single monomorphization.
    pub category: FunctionCategory,
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FunctionCategory {
    FullyCovered,
    PartiallyCovered,
    FullyUncovered,
}

impl FunctionCategory {
    /// Name used in the HTML class attribute and in the report filenames.
    pub fn css_class(self) -> &'static str {
        match self {
            FunctionCategory::FullyCovered => "fully",
            FunctionCategory::PartiallyCovered => "partial",
            FunctionCategory::FullyUncovered => "uncovered",
        }
    }

    /// Name shown to whoever is reading the report.
    pub fn label(self) -> &'static str {
        match self {
            FunctionCategory::FullyCovered => "Fully Covered",
            FunctionCategory::PartiallyCovered => "Partially Covered",
            FunctionCategory::FullyUncovered => "Fully Uncovered",
        }
    }
}

/// Everything the report needs, so it can be built without going back to the
/// llvm-cov output.
#[derive(Serialize, Deserialize)]
pub struct CoverageData {
    pub functions: Vec<FunctionReport>,
    /// The text of every source file these functions came from, keyed by the
    /// path that appears in the coverage data.
    ///
    /// Read once here so that building the report never has to go to disk.
    pub sources: HashMap<String, Vec<String>>,
}

/// Decide whether a function is fully covered, partly covered, or not covered.
///
/// LLVM does not track every line, so only the lines it does track count here.
/// If all of them ran the function is fully covered, if none of them ran it is
/// uncovered, and anything in between is partly covered.
///
/// A function with no tracked lines counts as covered, since there is nothing
/// in it that could have been missed.
fn categorize(line_counts: &[Option<u64>]) -> FunctionCategory {
    let tracked: Vec<u64> = line_counts.iter().filter_map(|c| *c).collect();
    if tracked.is_empty() {
        FunctionCategory::FullyCovered
    } else if tracked.iter().all(|&c| c > 0) {
        FunctionCategory::FullyCovered
    } else if tracked.iter().all(|&c| c == 0) {
        FunctionCategory::FullyUncovered
    } else {
        FunctionCategory::PartiallyCovered
    }
}

/// Read the llvm-cov JSON and merge it down to one entry per function.
pub fn process(json_text: &str, src_root: &Path) -> Result<CoverageData> {
    let export: Export =
        serde_json::from_str(json_text).context("failed to parse llvm-cov JSON")?;

    let functions = export.data.into_iter().flat_map(|d| d.functions).collect::<Vec<_>>();
    eprintln!("{} functions loaded", functions.len());

    let mut source_cache: HashMap<String, Vec<String>> = HashMap::new();

    let mut reports: Vec<FunctionReport> = vec![];

    for func in &functions {
        let demangled = format!("{:#}", demangle(&func.name));

        // Only compiler crates. The name either starts with `rustc_`, or has
        // it just inside a leading `<` for an impl method.
        if !demangled.starts_with("rustc") && !demangled.contains("<rustc") {
            continue;
        }

        if func.filenames.is_empty() || func.regions.is_empty() {
            continue;
        }

        // find the primary source file (first one in the compiler/ tree)
        let filename = match func.filenames.iter().find(|f| f.contains("/compiler/")) {
            Some(f) => f.clone(),
            None => func.filenames[0].clone(),
        };

        // figure out overall line span from all regions
        let mut line_start = usize::MAX;
        let mut line_end = 0usize;
        for region in &func.regions {
            if region.len() < 5 {
                continue;
            }
            let rs = region[0].as_u64().unwrap_or(0) as usize;
            let re = region[2].as_u64().unwrap_or(0) as usize;
            if rs > 0 && rs < line_start {
                line_start = rs;
            }
            if re > line_end {
                line_end = re;
            }
        }
        if line_start == usize::MAX || line_end == 0 || line_start > line_end {
            continue;
        }

        // A region is [line_start, col_start, line_end, col_end, count, file_id,
        // expanded_file_id, kind], and only kind 0 is real code.
        //
        // Regions nest. An outer one counts a whole block, an inner one counts a
        // branch inside that block, so the innermost region covering a line is
        // the one that says whether the line ran. Pick the smallest span, and
        // the later start when two spans are the same size.
        let mut region_tightness: HashMap<usize, (usize, std::cmp::Reverse<usize>, u64)> =
            HashMap::new();
        for region in &func.regions {
            if region.len() < 8 {
                continue;
            }
            let kind = region[7].as_u64().unwrap_or(0);
            if kind != 0 {
                continue;
            }
            let rs = region[0].as_u64().unwrap_or(0) as usize;
            let re = region[2].as_u64().unwrap_or(0) as usize;
            let count = region[4].as_u64().unwrap_or(0);
            let span = re.saturating_sub(rs);
            for line in rs..=re {
                let key = (span, std::cmp::Reverse(rs), count);
                let entry = region_tightness.entry(line).or_insert(key);
                // prefer tighter (smaller span, then later start)
                if (span, std::cmp::Reverse(rs)) < (entry.0, entry.1) {
                    *entry = key;
                }
            }
        }
        let region_counts: HashMap<usize, u64> =
            region_tightness.into_iter().map(|(line, (_, _, count))| (line, count)).collect();

        if !source_cache.contains_key(&filename) {
            let resolved = resolve_source_path(&filename, src_root);
            let lines = match resolved.and_then(|p| std::fs::read_to_string(&p).ok()) {
                Some(text) => text.lines().map(|l| l.to_string()).collect::<Vec<_>>(),
                None => vec![],
            };
            source_cache.insert(filename.clone(), lines);
        }

        // `bug!` and `span_bug!` lines are meant never to run, so reporting
        // them as uncovered would only be noise.
        let source_lines = source_cache.get(&filename).cloned().unwrap_or_default();
        let mut line_counts: Vec<Option<u64>> = (line_start..=line_end)
            .map(|lineno| {
                let src =
                    source_lines.get(lineno.saturating_sub(1)).map(|s| s.trim()).unwrap_or("");
                if src.starts_with("bug!") || src.starts_with("span_bug!") {
                    return None;
                }
                region_counts.get(&lineno).copied()
            })
            .collect();

        // LLVM hangs branch-not-taken counts off closing braces, so a brace can
        // show as uncovered while the body above it clearly ran. Give it the
        // count of the last line that did run instead.
        let mut last_covered_count: Option<u64> = None;
        for i in 0..line_counts.len() {
            let lineno = line_start + i;
            let src = source_lines.get(lineno.saturating_sub(1)).map(|s| s.trim()).unwrap_or("");
            let is_closing =
                src == "}" || src == "};" || src == "}," || src == "});" || src == "})";
            match line_counts[i] {
                Some(c) if c > 0 => {
                    last_covered_count = Some(c);
                }
                Some(0) if is_closing => {
                    if let Some(c) = last_covered_count {
                        line_counts[i] = Some(c);
                    }
                }
                _ => {}
            }
        }

        // FIXME: a match arm's pattern shows as untracked even when its body
        // never ran. Marking those uncovered needs more than a lookahead, the
        // naive version marked far too many lines.

        reports.push(FunctionReport {
            id: 0,
            demangled,
            filename,
            line_start,
            category: categorize(&line_counts),
            line_counts,
        });
    }

    eprintln!("{} compiler functions processed (before merging monomorphizations)", reports.len());

    let reports = merge_monomorphizations(reports);
    eprintln!("{} functions after merging monomorphizations", reports.len());

    let mut reports = merge_closures(reports);
    eprintln!("{} functions after merging closures into parents", reports.len());

    // Left until now because merging changes the counts, and the ids have to
    // match the order the source shards get written in.
    for (id, report) in reports.iter_mut().enumerate() {
        report.id = id;
        report.category = categorize(&report.line_counts);
    }

    Ok(CoverageData { functions: reports, sources: source_cache })
}

/// Work out where a source file actually lives on this machine.
///
/// Paths in the coverage data are full paths from whichever machine built the
/// compiler, something like `/home/someone/rust/compiler/rustc_abi/src/x.rs`.
/// Run the report anywhere else and that path does not exist.
///
/// Everything from `/compiler/` onwards is the same in any checkout though, so
/// take that part and join it onto the checkout we were given.
///
/// Gives back `None` if the file still cannot be found. The function is then
/// still listed in the report, just with no source to show.
pub fn resolve_source_path(filename: &str, src_root: &Path) -> Option<PathBuf> {
    if let Some(idx) = filename.find("/compiler/") {
        let rel = &filename[idx + 1..];
        let candidate = src_root.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let p = PathBuf::from(filename);
    if p.exists() { Some(p) } else { None }
}

/// Add up the separate copies LLVM made of each generic function.
///
/// A generic function is compiled once for every set of types it is used with,
/// and LLVM reports each copy on its own. Two copies starting on the same line
/// of the same file are the same function, so their counts get added together.
///
/// This matters because different copies take different branches. A line that
/// only one copy ever reached still counts as covered.
fn merge_monomorphizations(reports: Vec<FunctionReport>) -> Vec<FunctionReport> {
    let mut groups: std::collections::BTreeMap<(String, usize), FunctionReport> =
        std::collections::BTreeMap::new();

    for report in reports {
        // Where a function lives never changes between monomorphizations,
        // even though its demangled name does (it carries the concrete
        // types). File and line is the only identity that's actually stable.
        let key = (report.filename.clone(), report.line_start);
        match groups.get_mut(&key) {
            None => {
                groups.insert(key, report);
            }
            Some(existing) => {
                // An untracked line stays untracked, anything else adds up.
                for (i, count) in report.line_counts.iter().enumerate() {
                    if let Some(existing_count) = existing.line_counts.get_mut(i) {
                        *existing_count = match (*existing_count, *count) {
                            (Some(a), Some(b)) => Some(a.saturating_add(b)),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                    }
                }
                // Keep the shortest name, it carries the least type noise.
                if report.demangled.len() < existing.demangled.len() {
                    existing.demangled = report.demangled;
                }
            }
        }
    }

    groups.into_values().collect()
}

#[cfg(test)]
fn make_report(
    demangled: &str,
    filename: &str,
    line_start: usize,
    line_counts: Vec<Option<u64>>,
) -> FunctionReport {
    FunctionReport {
        id: 0,
        demangled: demangled.to_string(),
        filename: filename.to_string(),
        line_start,
        category: categorize(&line_counts),
        line_counts,
    }
}

/// Take one `::{closure#N}` off the end of a name, if there is one.
///
/// Closures are named after the function holding them, so `foo::{closure#0}`
/// gives back `foo`. A closure inside another closure only loses one level per
/// call, so `foo::{closure#0}::{closure#1}` gives `foo::{closure#0}`. Keep
/// calling until it returns `None` to get back to the real function.
fn closure_parent(name: &str) -> Option<&str> {
    let idx = name.rfind("::{closure")?;
    Some(&name[..idx])
}

/// Move each closure's coverage into the function it is written inside.
///
/// LLVM reports a closure as a function of its own. To anyone reading the
/// report a closure is just part of the function around it, so leaving them
/// separate would split one function into several entries.
fn merge_closures(reports: Vec<FunctionReport>) -> Vec<FunctionReport> {
    let mut groups: std::collections::BTreeMap<(String, String), FunctionReport> =
        std::collections::BTreeMap::new();

    for report in reports {
        // Closures nest, so keep stripping until there is nothing left to strip.
        let mut root = report.demangled.as_str();
        while let Some(parent) = closure_parent(root) {
            root = parent;
        }
        // Same reasoning as merge_monomorphizations: root is the real
        // function's name once every closure suffix is stripped off, and
        // that's what identifies it, not the raw demangled name.
        let key = (report.filename.clone(), root.to_string());

        let root_owned = root.to_string();
        match groups.get_mut(&key) {
            None => {
                let mut r = report;
                r.demangled = root_owned;
                groups.insert(key, r);
            }
            Some(existing) => {
                // Each side counts from its own `line_start`, and a closure
                // hardly ever starts where its parent does. Line the two up by
                // real line number, not by position in the vec.
                let new_line_start = existing.line_start.min(report.line_start);
                let existing_line_end = existing.line_start + existing.line_counts.len();
                let report_line_end = report.line_start + report.line_counts.len();
                let new_line_end = existing_line_end.max(report_line_end);

                let mut merged: Vec<Option<u64>> =
                    vec![None; new_line_end.saturating_sub(new_line_start)];

                let place =
                    |line_start: usize, counts: &[Option<u64>], merged: &mut Vec<Option<u64>>| {
                        for (i, count) in counts.iter().enumerate() {
                            let lineno = line_start + i;
                            let idx = lineno - new_line_start;
                            merged[idx] = match (merged[idx], *count) {
                                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                                (Some(a), None) => Some(a),
                                (None, Some(b)) => Some(b),
                                (None, None) => None,
                            };
                        }
                    };
                place(existing.line_start, &existing.line_counts, &mut merged);
                place(report.line_start, &report.line_counts, &mut merged);

                existing.line_start = new_line_start;
                existing.line_counts = merged;
            }
        }
    }

    groups.into_values().collect()
}

#[cfg(test)]
mod tests;
