//! Checks that all Flunt files have messages in alphabetical order

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use regex::Regex;

use crate::diagnostics::{CheckId, DiagCtx, RunningCheck};
use crate::walk::{filter_dirs, walk};

fn message() -> &'static Regex {
    static_regex!(r#"(?m)^([a-zA-Z0-9_]+)\s*=\s*"#)
}

fn is_fluent(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "ftl")
}

fn check_alphabetic(
    filename: &str,
    fluent: &str,
    check: &mut RunningCheck,
    all_defined_msgs: &mut HashMap<String, String>,
) {
    let mut matches = message().captures_iter(fluent).peekable();
    while let Some(m) = matches.next() {
        let name = m.get(1).unwrap();
        if let Some(defined_filename) = all_defined_msgs.get(name.as_str()) {
            check.error(format!(
                "{filename}: message `{}` is already defined in {defined_filename}",
                name.as_str(),
            ));
        }

        all_defined_msgs.insert(name.as_str().to_owned(), filename.to_owned());

        if let Some(next) = matches.peek() {
            let next = next.get(1).unwrap();
            if name.as_str() > next.as_str() {
                check.error(format!(
                    "{filename}: message `{}` appears before `{}`, but is alphabetically later than it
run `./x.py test tidy --bless` to sort the file correctly",
                    name.as_str(),
                    next.as_str()
                ));
            }
        } else {
            break;
        }
    }
}

fn sort_messages(
    filename: &str,
    fluent: &str,
    check: &mut RunningCheck,
    all_defined_msgs: &mut HashMap<String, String>,
) -> String {
    let mut chunks = vec![];
    let mut cur = String::new();
    for line in fluent.lines() {
        if let Some(name) = message().find(line) {
            if let Some(defined_filename) = all_defined_msgs.get(name.as_str()) {
                check.error(format!(
                    "{filename}: message `{}` is already defined in {defined_filename}",
                    name.as_str(),
                ));
            }

            all_defined_msgs.insert(name.as_str().to_owned(), filename.to_owned());
            chunks.push(std::mem::take(&mut cur));
        }

        cur += line;
        cur.push('\n');
    }
    chunks.push(cur);
    chunks.sort();
    let mut out = chunks.join("");
    out = out.trim().to_string();
    out.push('\n');
    out
}

pub fn check(path: &Path, bless: bool, diag_ctx: DiagCtx) {
    let mut check = diag_ctx.start_check(CheckId::new("fluent_alphabetical").path(path));

    let mut all_defined_msgs = HashMap::new();
    walk(
        path,
        |path, is_dir| filter_dirs(path) || (!is_dir && !is_fluent(path)),
        &mut |ent, contents| {
            if bless {
                let sorted = sort_messages(
                    ent.path().to_str().unwrap(),
                    contents,
                    &mut check,
                    &mut all_defined_msgs,
                );
                if sorted != contents {
                    let mut f =
                        OpenOptions::new().write(true).truncate(true).open(ent.path()).unwrap();
                    f.write_all(sorted.as_bytes()).unwrap();
                }
            } else {
                check_alphabetic(
                    ent.path().to_str().unwrap(),
                    contents,
                    &mut check,
                    &mut all_defined_msgs,
                );
            }
        },
    );

    crate::fluent_used::check(path, all_defined_msgs, diag_ctx);
}
