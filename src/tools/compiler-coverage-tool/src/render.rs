//! Writes the HTML report.
//!
//! The report is a small front page, then one page each for the functions that
//! are fully covered, partly covered, and not covered at all. Putting the whole
//! compiler on one page came to tens of megabytes, which browsers struggle to
//! open.
//!
//! Source code is not written into the pages either. It goes into separate JSON
//! files that a page only fetches when someone expands a function.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use askama::Template;
use serde::Serialize;

use crate::pct;
use crate::transform::{FunctionCategory, FunctionReport};

#[derive(Serialize)]
struct SourceLine<'a> {
    #[serde(rename = "n")]
    lineno: usize,
    #[serde(rename = "c")]
    class: &'static str,
    #[serde(rename = "t")]
    text: &'a str,
}

const SHARD_COUNT: usize = 16;

const REPORT_CSS: &str = include_str!("../static/report.css");
const SEARCH_JS: &str = include_str!("../static/search.js");
const SOURCE_LOADER_JS: &str = include_str!("../static/source-loader.js");

const CSS_FILE: &str = "report.css";
const SEARCH_JS_FILE: &str = "search.js";
const SOURCE_LOADER_JS_FILE: &str = "source-loader.js";

/// Put the stylesheet and scripts next to the report that links them.
pub fn write_static_assets(out_dir: &Path) -> Result<()> {
    for (name, contents) in [
        (CSS_FILE, REPORT_CSS),
        (SEARCH_JS_FILE, SEARCH_JS),
        (SOURCE_LOADER_JS_FILE, SOURCE_LOADER_JS),
    ] {
        let path = out_dir.join(name);
        std::fs::write(&path, contents)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

/// Write every function's source lines out across `SHARD_COUNT` JSON files.
///
/// Splitting them up means a page only downloads the one small file holding
/// the function someone just expanded, rather than every function's source at
/// once. Putting the source directly into the pages came to about 77MB.
pub fn write_source_shards(
    functions: &[FunctionReport],
    sources: &HashMap<String, Vec<String>>,
    shard_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(shard_dir)
        .with_context(|| format!("failed to create {}", shard_dir.display()))?;

    let mut shards: Vec<HashMap<usize, Vec<SourceLine<'_>>>> =
        (0..SHARD_COUNT).map(|_| HashMap::new()).collect();

    for report in functions {
        let source_lines = sources.get(&report.filename);
        let lines: Vec<SourceLine<'_>> = report
            .line_counts
            .iter()
            .enumerate()
            .map(|(i, count)| {
                let lineno = report.line_start + i;
                let class = match count {
                    Some(n) if *n > 0 => "c",
                    Some(_) => "u",
                    None => "i",
                };
                let text = source_lines
                    .and_then(|ls| ls.get(lineno.saturating_sub(1)))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                SourceLine { lineno, class, text }
            })
            .collect();
        shards[report.id % SHARD_COUNT].insert(report.id, lines);
    }

    for (i, shard) in shards.iter().enumerate() {
        let shard_path = shard_dir.join(format!("shard-{i}.json"));
        let json = serde_json::to_string(shard)
            .with_context(|| format!("failed to serialize shard {i}"))?;
        std::fs::write(&shard_path, json)
            .with_context(|| format!("failed to write {}", shard_path.display()))?;
    }

    Ok(())
}

/// The names of the files that make up one report.
///
/// `base_name` is the output filename without its extension, so `report` gives
/// `report.html`, `report_uncovered.html` and so on. Keeping them together
/// means the names are only decided in one place.
pub struct ReportPaths {
    pub index: String,
    pub fully_covered: String,
    pub partially_covered: String,
    pub uncovered: String,
}

pub fn report_paths(base_name: &str) -> ReportPaths {
    ReportPaths {
        index: format!("{base_name}.html"),
        fully_covered: format!("{base_name}_fully-covered.html"),
        partially_covered: format!("{base_name}_partially-covered.html"),
        uncovered: format!("{base_name}_uncovered.html"),
    }
}

impl ReportPaths {
    /// The page a category's functions get written to.
    pub fn for_category(&self, category: FunctionCategory) -> &str {
        match category {
            FunctionCategory::FullyCovered => &self.fully_covered,
            FunctionCategory::PartiallyCovered => &self.partially_covered,
            FunctionCategory::FullyUncovered => &self.uncovered,
        }
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate<'a> {
    css_file: &'static str,
    overall_pct: String,
    covered_lines_total: usize,
    tracked_lines_total: usize,
    uncovered_path: &'a str,
    partial_path: &'a str,
    fully_path: &'a str,
    uncovered_count: usize,
    partial_count: usize,
    fully_count: usize,
}

/// The page you land on: overall numbers, and links to the three categories.
pub fn render_index(
    fully_count: usize,
    partial_count: usize,
    uncovered_count: usize,
    covered_lines_total: usize,
    tracked_lines_total: usize,
    paths: &ReportPaths,
) -> Result<String> {
    let template = IndexTemplate {
        css_file: CSS_FILE,
        overall_pct: format!("{:.2}", pct(covered_lines_total, tracked_lines_total)),
        covered_lines_total,
        tracked_lines_total,
        uncovered_path: &paths.uncovered,
        partial_path: &paths.partially_covered,
        fully_path: &paths.fully_covered,
        uncovered_count,
        partial_count,
        fully_count,
    };
    Ok(template.render()?)
}

#[derive(Template)]
#[template(path = "category.html")]
struct CategoryTemplate<'a> {
    css_file: &'static str,
    search_js_file: &'static str,
    source_loader_js_file: &'static str,
    label: &'a str,
    this_count: usize,
    cat_class: &'a str,
    index_path: &'a str,
    uncovered_path: &'a str,
    partial_path: &'a str,
    fully_path: &'a str,
    uncovered_active: bool,
    partial_active: bool,
    fully_active: bool,
    shard_count: usize,
    shard_dir_name: &'a str,
    crates: Vec<CrateGroup>,
}

/// One category's page, with its functions grouped by crate and then file.
pub fn render_category_page(
    functions: &[&FunctionReport],
    category: FunctionCategory,
    shard_dir_name: &str,
    paths: &ReportPaths,
) -> Result<String> {
    let cat_class = category.css_class();
    let template = CategoryTemplate {
        css_file: CSS_FILE,
        search_js_file: SEARCH_JS_FILE,
        source_loader_js_file: SOURCE_LOADER_JS_FILE,
        label: category.label(),
        this_count: functions.len(),
        cat_class,
        index_path: &paths.index,
        uncovered_path: &paths.uncovered,
        partial_path: &paths.partially_covered,
        fully_path: &paths.fully_covered,
        uncovered_active: cat_class == FunctionCategory::FullyUncovered.css_class(),
        partial_active: cat_class == FunctionCategory::PartiallyCovered.css_class(),
        fully_active: cat_class == FunctionCategory::FullyCovered.css_class(),
        shard_count: SHARD_COUNT,
        shard_dir_name,
        crates: group_by_crate_and_file(functions),
    };
    Ok(template.render()?)
}

struct FnRow {
    fn_id: usize,
    fn_name: String,
    badge_class: &'static str,
    badge_text: String,
    file_line: String,
}

struct FileGroup {
    file: String,
    count: usize,
    fns: Vec<FnRow>,
}

struct CrateGroup {
    krate: String,
    count: usize,
    files: Vec<FileGroup>,
}

/// Build the crate to file to function tree that the template walks.
fn group_by_crate_and_file(functions: &[&FunctionReport]) -> Vec<CrateGroup> {
    let mut seen_crates: Vec<String> = vec![];
    for report in functions {
        let krate = crate_name(&report.filename);
        if !seen_crates.contains(&krate) {
            seen_crates.push(krate);
        }
    }

    seen_crates
        .into_iter()
        .map(|krate| {
            let krate_fns: Vec<&&FunctionReport> =
                functions.iter().filter(|r| crate_name(&r.filename) == krate).collect();

            let mut seen_files: Vec<String> = vec![];
            for report in &krate_fns {
                let f = file_path_in_crate(&report.filename);
                if !seen_files.contains(&f) {
                    seen_files.push(f);
                }
            }

            let files = seen_files
                .into_iter()
                .map(|file| {
                    let fns: Vec<FnRow> = krate_fns
                        .iter()
                        .filter(|r| file_path_in_crate(&r.filename) == file)
                        .map(|r| build_fn_row(r))
                        .collect();

                    FileGroup { count: fns.len(), file, fns }
                })
                .collect::<Vec<_>>();

            CrateGroup { count: krate_fns.len(), krate, files }
        })
        .collect()
}

/// Work out what the template needs to draw one function's row.
fn build_fn_row(report: &FunctionReport) -> FnRow {
    let short_filename = if let Some(idx) = report.filename.find("/compiler/") {
        &report.filename[idx + 1..]
    } else {
        &report.filename
    };

    let covered_lines = report.line_counts.iter().filter(|c| c.map_or(false, |n| n > 0)).count();
    let total_tracked = report.line_counts.iter().filter(|c| c.is_some()).count();
    let fn_pct = if total_tracked > 0 { pct(covered_lines, total_tracked) } else { 100.0 };

    let (badge_class, badge_text) = match report.category {
        FunctionCategory::FullyCovered => ("badge-fully", "100% covered".to_string()),
        FunctionCategory::PartiallyCovered => ("badge-partial", format!("{fn_pct:.0}% covered")),
        FunctionCategory::FullyUncovered => ("badge-uncovered", "0% covered".to_string()),
    };

    // FIXME: link this to the function's source on GitHub. Taken out for now
    // because the report is built from a local checkout, so any uncommitted
    // edit puts the line numbers out of step with what GitHub would show.
    FnRow {
        fn_id: report.id,
        fn_name: report.demangled.clone(),
        badge_class,
        badge_text,
        file_line: format!("{}:{}", short_filename, report.line_start),
    }
}

fn crate_name(filename: &str) -> String {
    if let Some(idx) = filename.find("/compiler/") {
        let after = &filename[idx + "/compiler/".len()..];
        return after.split('/').next().unwrap_or("unknown").to_string();
    }
    "unknown".to_string()
}

fn file_path_in_crate(filename: &str) -> String {
    if let Some(idx) = filename.find("/compiler/") {
        let after = &filename[idx + "/compiler/".len()..];
        if let Some(slash) = after.find('/') {
            return after[slash + 1..].to_string();
        }
    }
    filename.to_string()
}

#[cfg(test)]
mod tests;
