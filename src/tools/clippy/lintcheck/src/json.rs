use std::fs;
use std::path::Path;

use itertools::EitherOrBoth;
use serde::{Deserialize, Serialize};

use crate::ClippyWarning;

#[derive(Deserialize, Serialize)]
struct LintJson {
    lint: String,
    file_name: String,
    byte_pos: (u32, u32),
    rendered: String,
}

impl LintJson {
    fn key(&self) -> impl Ord + '_ {
        (self.file_name.as_str(), self.byte_pos, self.lint.as_str())
    }
}

/// Creates the log file output for [`crate::config::OutputFormat::Json`]
pub(crate) fn output(clippy_warnings: Vec<ClippyWarning>) -> String {
    let mut lints: Vec<LintJson> = clippy_warnings
        .into_iter()
        .map(|warning| {
            let span = warning.span();
            LintJson {
                file_name: span.file_name.clone(),
                byte_pos: (span.byte_start, span.byte_end),
                lint: warning.lint,
                rendered: warning.diag.rendered.unwrap(),
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

fn print_warnings(title: &str, warnings: &[LintJson]) {
    if warnings.is_empty() {
        return;
    }

    println!("### {title}");
    println!("```");
    for warning in warnings {
        print!("{}", warning.rendered);
    }
    println!("```");
}

fn print_changed_diff(changed: &[(LintJson, LintJson)]) {
    if changed.is_empty() {
        return;
    }

    println!("### Changed");
    println!("```diff");
    for (old, new) in changed {
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
    }
    println!("```");
}

pub(crate) fn diff(old_path: &Path, new_path: &Path) {
    let old_warnings = load_warnings(old_path);
    let new_warnings = load_warnings(new_path);

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for change in itertools::merge_join_by(old_warnings, new_warnings, |old, new| old.key().cmp(&new.key())) {
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

    print!(
        "{} added, {} removed, {} changed\n\n",
        added.len(),
        removed.len(),
        changed.len()
    );

    print_warnings("Added", &added);
    print_warnings("Removed", &removed);
    print_changed_diff(&changed);
}
