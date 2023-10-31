//! Checks that all Flunt files have messages in alphabetical order

use crate::walk::{filter_dirs, walk};
use std::{fs::OpenOptions, io::Write, path::Path};

use regex::Regex;

lazy_static::lazy_static! {
    static ref MESSAGE: Regex = Regex::new(r#"(?m)^([a-zA-Z0-9_]+)\s*=\s*"#).unwrap();
}

fn filter_fluent(path: &Path) -> bool {
    if let Some(ext) = path.extension() { ext.to_str() != Some("ftl") } else { true }
}

fn check_alphabetic(filename: &str, fluent: &str, bad: &mut bool) {
    let mut matches = MESSAGE.captures_iter(fluent).peekable();
    while let Some(m) = matches.next() {
        if let Some(next) = matches.peek() {
            let name = m.get(1).unwrap();
            let next = next.get(1).unwrap();
            if name.as_str() > next.as_str() {
                tidy_error!(
                    bad,
                    "{filename}: message `{}` appears before `{}`, but is alphabetically later than it
run `./x.py test tidy --bless` to sort the file correctly",
                    name.as_str(),
                    next.as_str()
                );
            }
        } else {
            break;
        }
    }
}

fn sort_messages(fluent: &str) -> String {
    let mut chunks = vec![];
    let mut cur = String::new();
    for line in fluent.lines() {
        if MESSAGE.is_match(line) {
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

pub fn check(path: &Path, bless: bool, bad: &mut bool) {
    walk(
        path,
        |path, is_dir| filter_dirs(path) || (!is_dir && filter_fluent(path)),
        &mut |ent, contents| {
            if bless {
                let sorted = sort_messages(contents);
                if sorted != contents {
                    let mut f =
                        OpenOptions::new().write(true).truncate(true).open(ent.path()).unwrap();
                    f.write(sorted.as_bytes()).unwrap();
                }
            } else {
                check_alphabetic(ent.path().to_str().unwrap(), contents, bad);
            }
        },
    );
}
