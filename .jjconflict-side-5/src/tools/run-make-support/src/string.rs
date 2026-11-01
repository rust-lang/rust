use std::path::Path;

use crate::fs;
use crate::path_helpers::{cwd, has_extension, shallow_find_files};

/// Gathers all files in the current working directory that have the extension `ext`, and counts
/// the number of lines within that contain a match with the regex pattern `re`.
pub fn count_regex_matches_in_files_with_extension(re: &regex::Regex, ext: &str) -> usize {
    let fetched_files = shallow_find_files(cwd(), |path| has_extension(path, ext));

    let mut count = 0;
    for file in fetched_files {
        let content = fs::read_to_string(file);
        count += content.lines().filter(|line| re.is_match(&line)).count();
    }

    count
}

/// Read the contents of a file that cannot simply be read by
/// [`read_to_string`][crate::fs::read_to_string], due to invalid UTF-8 data, then assert
/// that it contains `expected`.
#[track_caller]
pub fn invalid_utf8_contains<P: AsRef<Path>, S: AsRef<str>>(path: P, expected: S) {
    let buffer = fs::read(path.as_ref());
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
/// [`read_to_string`][crate::fs::read_to_string], due to invalid UTF-8 data, then assert
/// that it does not contain `expected`.
#[track_caller]
pub fn invalid_utf8_not_contains<P: AsRef<Path>, S: AsRef<str>>(path: P, expected: S) {
    let buffer = fs::read(path.as_ref());
    let expected = expected.as_ref();
    if String::from_utf8_lossy(&buffer).contains(expected) {
        eprintln!("=== FILE CONTENTS (LOSSY) ===");
        eprintln!("{}", String::from_utf8_lossy(&buffer));
        eprintln!("=== SPECIFIED TEXT ===");
        eprintln!("{}", expected);
        panic!("specified text was unexpectedly found in file");
    }
}
