//! Collection of assertions and assertion-related helpers.

use std::path::{Path, PathBuf};
use std::panic;

use crate::fs_wrapper;
use crate::path_helpers::cwd;

/// Browse the directory `path` non-recursively and return all files which respect the parameters
/// outlined by `closure`.
#[track_caller]
pub fn shallow_find_files<P: AsRef<Path>, F: Fn(&PathBuf) -> bool>(
    path: P,
    filter: F,
) -> Vec<PathBuf> {
    let mut matching_files = Vec::new();
    for entry in fs_wrapper::read_dir(path) {
        let entry = entry.expect("failed to read directory entry.");
        let path = entry.path();

        if path.is_file() && filter(&path) {
            matching_files.push(path);
        }
    }
    matching_files
}

/// Returns true if the filename at `path` starts with `prefix`.
pub fn has_prefix<P: AsRef<Path>>(path: P, prefix: &str) -> bool {
    path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().starts_with(prefix))
}

/// Returns true if the filename at `path` has the extension `extension`.
pub fn has_extension<P: AsRef<Path>>(path: P, extension: &str) -> bool {
    path.as_ref().extension().is_some_and(|ext| ext == extension)
}

/// Returns true if the filename at `path` does not contain `expected`.
pub fn not_contains<P: AsRef<Path>>(path: P, expected: &str) -> bool {
    !path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().contains(expected))
}

/// Returns true if the filename at `path` is not in `expected`.
pub fn filename_not_in_denylist<P: AsRef<Path>, V: AsRef<[String]>>(path: P, expected: V) -> bool {
    let expected = expected.as_ref();
    path.as_ref()
        .file_name()
        .is_some_and(|name| !expected.contains(&name.to_str().unwrap().to_owned()))
}

/// Returns true if the filename at `path` ends with `suffix`.
pub fn has_suffix<P: AsRef<Path>>(path: P, suffix: &str) -> bool {
    path.as_ref().file_name().is_some_and(|name| name.to_str().unwrap().ends_with(suffix))
}

/// Gathers all files in the current working directory that have the extension `ext`, and counts
/// the number of lines within that contain a match with the regex pattern `re`.
pub fn count_regex_matches_in_files_with_extension(re: &regex::Regex, ext: &str) -> usize {
    let fetched_files = shallow_find_files(cwd(), |path| has_extension(path, ext));

    let mut count = 0;
    for file in fetched_files {
        let content = fs_wrapper::read_to_string(file);
        count += content.lines().filter(|line| re.is_match(&line)).count();
    }

    count
}

/// Read the contents of a file that cannot simply be read by
/// [`read_to_string`][crate::fs_wrapper::read_to_string], due to invalid UTF-8 data, then assert
/// that it contains `expected`.
#[track_caller]
pub fn invalid_utf8_contains<P: AsRef<Path>, S: AsRef<str>>(path: P, expected: S) {
    let buffer = fs_wrapper::read(path.as_ref());
    let expected = expected.as_ref();
    if !String::from_utf8_lossy(&buffer).contains(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&buffer));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was not found in file");
    }
}

/// Read the contents of a file that cannot simply be read by
/// [`read_to_string`][crate::fs_wrapper::read_to_string], due to invalid UTF-8 data, then assert
/// that it does not contain `expected`.
#[track_caller]
pub fn invalid_utf8_not_contains<P: AsRef<Path>, S: AsRef<str>>(path: P, expected: S) {
    let buffer = fs_wrapper::read(path.as_ref());
    let expected = expected.as_ref();
    if String::from_utf8_lossy(&buffer).contains(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&buffer));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was unexpectedly found in file");
    }
}

/// Assert that `actual` is equal to `expected`.
#[track_caller]
pub fn assert_equals<A: AsRef<str>, E: AsRef<str>>(actual: A, expected: E) {
    let actual = actual.as_ref();
    let expected = expected.as_ref();
    if actual != expected {
        eprintln!("=== ACTUAL TEXT ===");
        eprintln!("{}", actual);
        eprintln!("=== EXPECTED ===");
        eprintln!("{}", expected);
        panic!("expected text was not found in actual text");
    }
}

/// Assert that `haystack` contains `needle`.
#[track_caller]
pub fn assert_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    if !haystack.contains(needle) {
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", needle);
        panic!("needle was not found in haystack");
    }
}

/// Assert that `haystack` does not contain `needle`.
#[track_caller]
pub fn assert_not_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    if haystack.contains(needle) {
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", needle);
        panic!("needle was unexpectedly found in haystack");
    }
}
