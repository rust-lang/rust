//! JSON output and comparison functionality for Clippy warnings.
//!
//! This module handles serialization of Clippy warnings to JSON format,
//! loading warnings from JSON files, and generating human-readable diffs
//! between different linting runs.

use std::path::{Path, PathBuf};
use std::{fmt, fs};

use itertools::{EitherOrBoth, Itertools};
use serde::{Deserialize, Serialize};

use crate::ClippyWarning;

/// This is the total number. 300 warnings results in 100 messages per section.
const DEFAULT_LIMIT_PER_LINT: usize = 300;
/// Target for total warnings to display across all lints when truncating output.
const TRUNCATION_TOTAL_TARGET: usize = 1000;

#[derive(Debug, Deserialize, Serialize)]
struct LintJson {
    /// The lint name e.g. `clippy::bytes_nth`
    name: String,
    /// The filename and line number e.g. `anyhow-1.0.86/src/error.rs:42`
    file_line: String,
    file_url: String,
    rendered: String,
}

impl LintJson {
    fn key(&self) -> impl Ord + '_ {
        (self.name.as_str(), self.file_line.as_str())
    }

    /// Formats the warning information with an action verb for display.
    fn info_text(&self, action: &str) -> String {
        format!("{action} `{}` at [`{}`]({})", self.name, self.file_line, self.file_url)
    }
}

#[derive(Debug, Serialize)]
struct SummaryRow {
    name: String,
    added: usize,
    removed: usize,
    changed: usize,
}

#[derive(Debug, Serialize)]
struct Summary(Vec<SummaryRow>);

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "\
| Lint | Added | Removed | Changed |
| ---- | ----: | ------: | ------: |
",
        )?;

        for SummaryRow {
            name,
            added,
            changed,
            removed,
        } in &self.0
        {
            let html_id = to_html_id(name);
            writeln!(f, "| [`{name}`](#{html_id}) | {added} | {removed} | {changed} |")?;
        }

        Ok(())
    }
}

impl Summary {
    fn new(lints: &[LintWarnings]) -> Self {
        Summary(
            lints
                .iter()
                .map(|lint| SummaryRow {
                    name: lint.name.clone(),
                    added: lint.added.len(),
                    removed: lint.removed.len(),
                    changed: lint.changed.len(),
                })
                .collect(),
        )
    }
}

/// Creates the log file output for [`crate::config::OutputFormat::Json`]
pub(crate) fn output(clippy_warnings: Vec<ClippyWarning>) -> String {
    let mut lints: Vec<LintJson> = clippy_warnings
        .into_iter()
        .map(|warning| {
            let span = warning.span();
            let file_name = span
                .file_name
                .strip_prefix("target/lintcheck/sources/")
                .unwrap_or(&span.file_name);
            let file_line = format!("{file_name}:{}", span.line_start);
            LintJson {
                name: warning.name,
                file_line,
                file_url: warning.url,
                rendered: warning.diag.rendered.unwrap().trim().to_string(),
            }
        })
        .collect();
    lints.sort_by(|a, b| a.key().cmp(&b.key()));
    serde_json::to_string(&lints).unwrap()
}

/// Loads lint warnings from a JSON file at the given path.
fn load_warnings(path: &Path) -> Vec<LintJson> {
    let file = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    serde_json::from_slice(&file).unwrap_or_else(|e| panic!("failed to deserialize {}: {e}", path.display()))
}

/// Generates and prints a diff between two sets of lint warnings.
///
/// Compares warnings from `old_path` and `new_path`, then displays a summary table
/// and detailed information about added, removed, and changed warnings.
pub(crate) fn diff(old_path: &Path, new_path: &Path, truncate: bool, write_summary: Option<PathBuf>) {
    let old_warnings = load_warnings(old_path);
    let new_warnings = load_warnings(new_path);

    let mut lint_warnings = vec![];

    for (name, changes) in &itertools::merge_join_by(old_warnings, new_warnings, |old, new| old.key().cmp(&new.key()))
        .chunk_by(|change| change.as_ref().into_left().name.clone())
    {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();
        for change in changes {
            match change {
                EitherOrBoth::Both(old, new) => {
                    if old.rendered != new.rendered {
                        changed.push((old, new));
                    }
                },
                EitherOrBoth::Left(old) => removed.push(old),
                EitherOrBoth::Right(new) => added.push(new),
            }
        }

        if !added.is_empty() || !removed.is_empty() || !changed.is_empty() {
            lint_warnings.push(LintWarnings {
                name,
                added,
                removed,
                changed,
            });
        }
    }

    if lint_warnings.is_empty() {
        return;
    }

    let summary = Summary::new(&lint_warnings);
    if let Some(path) = write_summary {
        let json = serde_json::to_string(&summary).unwrap();
        fs::write(path, json).unwrap();
    }

    let truncate_after = if truncate {
        // Max 15 ensures that we at least have five messages per lint
        DEFAULT_LIMIT_PER_LINT
            .min(TRUNCATION_TOTAL_TARGET / lint_warnings.len())
            .max(15)
    } else {
        // No lint should ever each this number of lint emissions, so this is equivialent to
        // No truncation
        usize::MAX
    };

    println!("{summary}");
    for lint in lint_warnings {
        print_lint_warnings(&lint, truncate_after);
    }
}

/// Container for grouped lint warnings organized by status (added/removed/changed).
#[derive(Debug)]
struct LintWarnings {
    name: String,
    added: Vec<LintJson>,
    removed: Vec<LintJson>,
    changed: Vec<(LintJson, LintJson)>,
}

fn print_lint_warnings(lint: &LintWarnings, truncate_after: usize) {
    let name = &lint.name;
    let html_id = to_html_id(name);

    println!(r#"<h2 id="{html_id}"><code>{name}</code></h2>"#);
    println!();

    print!(
        r"{}, {}, {}",
        count_string(name, "added", lint.added.len()),
        count_string(name, "removed", lint.removed.len()),
        count_string(name, "changed", lint.changed.len()),
    );
    println!();

    print_warnings("Added", &lint.added, truncate_after / 3);
    print_warnings("Removed", &lint.removed, truncate_after / 3);
    print_changed_diff(&lint.changed, truncate_after / 3);
}

/// Prints a section of warnings with a header and formatted code blocks.
fn print_warnings(title: &str, warnings: &[LintJson], truncate_after: usize) {
    if warnings.is_empty() {
        return;
    }

    print_h3(&warnings[0].name, title);
    println!();

    let warnings = truncate(warnings, truncate_after);

    for warning in warnings {
        println!("{}", warning.info_text(title));
        println!();
        println!("```");
        println!("{}", warning.rendered);
        println!("```");
        println!();
    }
}

/// Prints a section of changed warnings with unified diff format.
fn print_changed_diff(changed: &[(LintJson, LintJson)], truncate_after: usize) {
    if changed.is_empty() {
        return;
    }

    print_h3(&changed[0].0.name, "Changed");
    println!();

    let changed = truncate(changed, truncate_after);

    for (old, new) in changed {
        println!("{}", new.info_text("Changed"));
        println!();
        println!("```diff");
        for change in diff::lines(&old.rendered, &new.rendered) {
            use diff::Result::{Both, Left, Right};

            match change {
                Both(unchanged, _) => {
                    println!(" {unchanged}");
                },
                Left(removed) => {
                    println!("-{removed}");
                },
                Right(added) => {
                    println!("+{added}");
                },
            }
        }
        println!("```");
    }
}

/// Truncates a list to a maximum number of items and prints a message about truncation.
fn truncate<T>(list: &[T], truncate_after: usize) -> &[T] {
    if list.len() > truncate_after {
        println!(
            "{} warnings have been truncated for this summary.",
            list.len() - truncate_after
        );
        println!();

        list.split_at(truncate_after).0
    } else {
        list
    }
}

fn print_h3(lint: &str, title: &str) {
    let html_id = to_html_id(lint);
    // We have to use HTML here to be able to manually add an id, GitHub doesn't add them automatically
    println!(r#"<h3 id="{html_id}-{title}">{title}</h3>"#);
}

/// Creates a custom ID allowed by GitHub, they must start with `user-content-` and cannot contain
/// `::`/`_`
fn to_html_id(lint_name: &str) -> String {
    lint_name.replace("clippy::", "user-content-").replace('_', "-")
}

/// This generates the `x added` string for the start of the job summery.
/// It linkifies them if possible to jump to the respective heading.
fn count_string(lint: &str, label: &str, count: usize) -> String {
    // Headlines are only added, if anything will be displayed under the headline.
    // We therefore only want to add links to them if they exist
    if count == 0 {
        format!("0 {label}")
    } else {
        let html_id = to_html_id(lint);
        format!("[{count} {label}](#{html_id}-{label})")
    }
}
