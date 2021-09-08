#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::assertions_on_constants)]

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
                .map(|s| format!("\t{}", s))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/*
Test for missing files.

Since rs files are alphabetically before stderr/stdout, we can sort by the full name
and iter in that order. If we've seen the file stem for the first time and it's not
a rust file, it means the rust file has to be missing.
*/
fn explore_directory(dir: &Path) -> Vec<String> {
    let mut missing_files: Vec<String> = Vec::new();
    let mut current_file = String::new();
    let mut files: Vec<DirEntry> = fs::read_dir(dir).unwrap().filter_map(Result::ok).collect();
    files.sort_by_key(std::fs::DirEntry::path);
    for entry in &files {
        let path = entry.path();
        if path.is_dir() {
            missing_files.extend(explore_directory(&path));
        } else {
            let file_stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            if let Some(ext) = path.extension() {
                match ext.to_str().unwrap() {
                    "rs" => current_file = file_stem.clone(),
                    "stderr" | "stdout" => {
                        if file_stem != current_file {
                            missing_files.push(path.to_str().unwrap().to_string());
                        }
                    },
                    _ => continue,
                };
            }
        }
    }
    missing_files
}
