use std::fs;
use std::path::Path;

use itertools::{EitherOrBoth, Itertools};
use serde::{Deserialize, Serialize};

use crate::ClippyWarning;

/// This is the total number. 300 warnings results in 100 messages per section.
const DEFAULT_LIMIT_PER_LINT: usize = 300;
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

    fn info_text(&self, action: &str) -> String {
        format!("{action} `{}` at [`{}`]({})", self.name, self.file_line, self.file_url)
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

fn load_warnings(path: &Path) -> Vec<LintJson> {
    let file = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    serde_json::from_slice(&file).unwrap_or_else(|e| panic!("failed to deserialize {}: {e}", path.display()))
}

pub(crate) fn diff(old_path: &Path, new_path: &Path, truncate: bool) {
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

    print_summary_table(&lint_warnings);
    println!();

    if lint_warnings.is_empty() {
        return;
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

    for lint in lint_warnings {
        print_lint_warnings(&lint, truncate_after);
    }
}

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

    // The additional anchor is added for non GH viewers that don't prefix ID's
    println!(r#"## `{name}` <a id="user-content-{html_id}"></a>"#);
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

fn print_summary_table(lints: &[LintWarnings]) {
    println!("| Lint                                       | Added   | Removed | Changed |");
    println!("| ------------------------------------------ | ------: | ------: | ------: |");

    for lint in lints {
        println!(
            "| {:<62} | {:>7} | {:>7} | {:>7} |",
            format!("[`{}`](#user-content-{})", lint.name, to_html_id(&lint.name)),
            lint.added.len(),
            lint.removed.len(),
            lint.changed.len()
        );
    }
}

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
    // We have to use HTML here to be able to manually add an id.
    println!(r#"### {title} <a id="user-content-{html_id}-{title}"></a>"#);
}

/// GitHub's markdown parsers doesn't like IDs with `::` and `_`. This simplifies
/// the lint name for the HTML ID.
fn to_html_id(lint_name: &str) -> String {
    lint_name.replace("clippy::", "").replace('_', "-")
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
        // GitHub's job summaries don't add HTML ids to headings. That's why we
        // manually have to add them. GitHub prefixes these manual ids with
        // `user-content-` and that's how we end up with these awesome links :D
        format!("[{count} {label}](#user-content-{html_id}-{label})")
    }
}
