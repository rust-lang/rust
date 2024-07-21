use std::fs;
use std::path::Path;

use itertools::EitherOrBoth;
use serde::{Deserialize, Serialize};

use crate::ClippyWarning;

#[derive(Deserialize, Serialize)]
struct LintJson {
    lint: String,
    krate: String,
    file_name: String,
    byte_pos: (u32, u32),
    file_link: String,
    rendered: String,
}

impl LintJson {
    fn key(&self) -> impl Ord + '_ {
        (self.file_name.as_str(), self.byte_pos, self.lint.as_str())
    }

    fn info_text(&self, action: &str) -> String {
        format!("{action} `{}` in `{}` at {}", self.lint, self.krate, self.file_link)
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
                krate: warning.krate,
                file_link: warning.url,
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

    //We have to use HTML here to be able to manually add an id.
    println!(r#"<h3 id="{title}">{title}</h3>"#);
    println!();
    for warning in warnings {
        println!("{}", warning.info_text(title));
        println!();
        println!("```");
        println!("{}", warning.rendered.trim_end());
        println!("```");
        println!();
    }
}

fn print_changed_diff(changed: &[(LintJson, LintJson)]) {
    if changed.is_empty() {
        return;
    }

    //We have to use HTML here to be able to manually add an id.
    println!(r#"<h3 id="changed">Changed</h3>"#);
    println!();
    for (old, new) in changed {
        println!("{}", new.info_text("Changed"));
        println!();
        println!("```diff");
        for change in diff::lines(old.rendered.trim_end(), new.rendered.trim_end()) {
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
        r##"{}, {}, {}"##,
        count_string("added", added.len()),
        count_string("removed", removed.len()),
        count_string("changed", changed.len()),
    );
    println!();
    println!();

    print_warnings("Added", &added);
    print_warnings("Removed", &removed);
    print_changed_diff(&changed);
}

/// This generates the `x added` string for the start of the job summery.
/// It linkifies them if possible to jump to the respective heading.
fn count_string(label: &str, count: usize) -> String {
    // Headlines are only added, if anything will be displayed under the headline.
    // We therefore only want to add links to them if they exist
    if count == 0 {
        format!("0 {label}")
    } else {
        // GitHub's job summaries don't add HTML ids to headings. That's why we
        // manually have to add them. GitHub prefixes these manual ids with
        // `user-content-` and that's how we end up with these awesome links :D
        format!("[{count} {label}](#user-content-{label})")
    }
}
