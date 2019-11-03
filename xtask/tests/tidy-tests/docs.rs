use std::fs;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use walkdir::{DirEntry, WalkDir};

use xtask::project_root;

fn is_exclude_dir(p: &Path) -> bool {
    // Test hopefully don't really need comments, and for assists we already
    // have special comments which are source of doc tests and user docs.
    let exclude_dirs = ["tests", "test_data", "assists"];
    let mut cur_path = p;
    while let Some(path) = cur_path.parent() {
        if exclude_dirs.iter().any(|dir| path.ends_with(dir)) {
            return true;
        }
        cur_path = path;
    }

    false
}

fn is_exclude_file(d: &DirEntry) -> bool {
    let file_names = ["tests.rs"];

    d.file_name().to_str().map(|f_n| file_names.iter().any(|name| *name == f_n)).unwrap_or(false)
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
}

#[test]
fn no_docs_comments() {
    let crates = project_root().join("crates");
    let iter = WalkDir::new(crates);
    let mut missing_docs = Vec::new();
    for f in iter.into_iter().filter_entry(|e| !is_hidden(e)) {
        let f = f.unwrap();
        if f.file_type().is_dir() {
            continue;
        }
        if f.path().extension().map(|it| it != "rs").unwrap_or(false) {
            continue;
        }
        if is_exclude_dir(f.path()) {
            continue;
        }
        if is_exclude_file(&f) {
            continue;
        }
        let mut reader = BufReader::new(fs::File::open(f.path()).unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if !line.starts_with("//!") {
            missing_docs.push(f.path().display().to_string());
        }
    }
    if !missing_docs.is_empty() {
        panic!(
            "\nMissing docs strings\n\n\
             modules:\n{}\n\n",
            missing_docs.join("\n")
        )
    }
}
