use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::hash::Hash;
use std::path::Path;

use crate::ClippyWarning;

/// Creates the log file output for [`crate::config::OutputFormat::Json`]
pub(crate) fn output(clippy_warnings: &[ClippyWarning]) -> String {
    serde_json::to_string(&clippy_warnings).unwrap()
}

fn load_warnings(path: &Path) -> Vec<ClippyWarning> {
    let file = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

    serde_json::from_slice(&file).unwrap_or_else(|e| panic!("failed to deserialize {}: {e}", path.display()))
}

/// Group warnings by their primary span location + lint name
fn create_map(warnings: &[ClippyWarning]) -> HashMap<impl Eq + Hash + '_, Vec<&ClippyWarning>> {
    let mut map = HashMap::<_, Vec<_>>::with_capacity(warnings.len());

    for warning in warnings {
        let span = warning.span();
        let key = (&warning.lint_type, &span.file_name, span.byte_start, span.byte_end);

        map.entry(key).or_default().push(warning);
    }

    map
}

fn print_warnings(title: &str, warnings: &[&ClippyWarning]) {
    if warnings.is_empty() {
        return;
    }

    println!("### {title}");
    println!("```");
    for warning in warnings {
        print!("{}", warning.diag);
    }
    println!("```");
}

fn print_changed_diff(changed: &[(&[&ClippyWarning], &[&ClippyWarning])]) {
    fn render(warnings: &[&ClippyWarning]) -> String {
        let mut rendered = String::new();
        for warning in warnings {
            write!(&mut rendered, "{}", warning.diag).unwrap();
        }
        rendered
    }

    if changed.is_empty() {
        return;
    }

    println!("### Changed");
    println!("```diff");
    for &(old, new) in changed {
        let old_rendered = render(old);
        let new_rendered = render(new);

        for change in diff::lines(&old_rendered, &new_rendered) {
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

    let old_map = create_map(&old_warnings);
    let new_map = create_map(&new_warnings);

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (key, new) in &new_map {
        if let Some(old) = old_map.get(key) {
            if old != new {
                changed.push((old.as_slice(), new.as_slice()));
            }
        } else {
            added.extend(new);
        }
    }

    for (key, old) in &old_map {
        if !new_map.contains_key(key) {
            removed.extend(old);
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
