#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::assertions_on_constants)]
#![cfg_attr(bootstrap, feature(path_file_prefix))]

use std::cmp::Ordering;
use std::ffi::OsStr;
use std::fs::{self, DirEntry};
use std::path::Path;

#[test]
fn test_missing_tests() {
    let missing_files = explore_directory(Path::new("./tests"));
    if !missing_files.is_empty() {
        assert!(
            false,
            "Didn't see a test file for the following files:\n\n{}\n",
            missing_files
                .iter()
                .map(|s| format!("\t{s}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

// Test for missing files.
fn explore_directory(dir: &Path) -> Vec<String> {
    let mut missing_files: Vec<String> = Vec::new();
    let mut current_file = String::new();
    let mut files: Vec<DirEntry> = fs::read_dir(dir).unwrap().filter_map(Result::ok).collect();
    files.sort_by(|x, y| {
        match x.path().file_prefix().cmp(&y.path().file_prefix()) {
            Ordering::Equal => (),
            ord => return ord,
        }
        // Sort rs files before the others if they share the same prefix. So when we see
        // the file prefix for the first time and it's not a rust file, it means the rust
        // file has to be missing.
        match (
            x.path().extension().and_then(OsStr::to_str),
            y.path().extension().and_then(OsStr::to_str),
        ) {
            (Some("rs" | "toml"), _) => Ordering::Less,
            (_, Some("rs" | "toml")) => Ordering::Greater,
            _ => Ordering::Equal,
        }
    });
    for entry in &files {
        let path = entry.path();
        if path.is_dir() {
            missing_files.extend(explore_directory(&path));
        } else {
            let file_prefix = path.file_prefix().unwrap().to_str().unwrap().to_string();
            if let Some(ext) = path.extension() {
                match ext.to_str().unwrap() {
                    "rs" | "toml" => current_file.clone_from(&file_prefix),
                    "stderr" | "stdout" => {
                        if file_prefix != current_file {
                            missing_files.push(path.to_str().unwrap().to_string());
                        }
                    },
                    _ => {},
                }
            }
        }
    }
    missing_files
}
