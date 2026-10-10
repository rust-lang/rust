//! Collection of assertions and assertion-related helpers.

#[cfg(test)]
mod tests;

use std::panic;
use std::path::Path;

use crate::{fs, regex};

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
        panic!("expected text does not match actual text");
    }
}

struct SearchDetails<'assertion_name, 'haystack, 'needle> {
    assertion_name: &'assertion_name str,
    haystack: &'haystack str,
    needle: &'needle str,
}

impl<'assertion_name, 'haystack, 'needle> SearchDetails<'assertion_name, 'haystack, 'needle> {
    fn dump(&self) {
        eprintln!("{}:", self.assertion_name);
        eprintln!("=== HAYSTACK ===");
        eprintln!("{}", self.haystack);
        eprintln!("=== NEEDLE ===");
        eprintln!("{}", self.needle);
    }
}

/// Assert that `haystack` contains `needle`.
#[track_caller]
pub fn assert_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    if !haystack.contains(needle) {
        SearchDetails { assertion_name: "assert_contains", haystack, needle }.dump();
        panic!("needle was not found in haystack");
    }
}

/// Assert that `haystack` does not contain `needle`.
#[track_caller]
pub fn assert_not_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    if haystack.contains(needle) {
        SearchDetails { assertion_name: "assert_not_contains", haystack, needle }.dump();
        panic!("needle was unexpectedly found in haystack");
    }
}

/// Assert that `haystack` contains the regex `needle`.
#[track_caller]
pub fn assert_contains_regex<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    let re = regex::Regex::new(needle).unwrap();
    if !re.is_match(haystack) {
        SearchDetails { assertion_name: "assert_contains_regex", haystack, needle }.dump();
        panic!("regex was not found in haystack");
    }
}

/// Assert that `haystack` does not contain the regex `needle`.
#[track_caller]
pub fn assert_not_contains_regex<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    let re = regex::Regex::new(needle).unwrap();
    if re.is_match(haystack) {
        SearchDetails { assertion_name: "assert_not_contains_regex", haystack, needle }.dump();
        panic!("regex was unexpectedly found in haystack");
    }
}

/// Assert that `haystack` contains regex `needle` an `expected_count` number of times.
#[track_caller]
pub fn assert_count_is<H: AsRef<str>, N: AsRef<str>>(
    expected_count: usize,
    haystack: H,
    needle: N,
) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();

    let actual_count = haystack.matches(needle).count();
    if expected_count != actual_count {
        let count_fmt = format!(
            "assert_count_is (expected_count = {expected_count}, actual_count = {actual_count})"
        );
        SearchDetails { assertion_name: &count_fmt, haystack, needle }.dump();
        panic!(
            "regex did not appear {expected_count} times in haystack (expected_count = \
            {expected_count}, actual_count = {actual_count})"
        );
    }
}

/// Assert that all files in `dir1` exist and have the same content in `dir2`
// FIXME(#135037): not robust against symlinks, lacks sanity test coverage.
pub fn assert_dirs_are_equal(dir1: impl AsRef<Path>, dir2: impl AsRef<Path>) {
    let dir2 = dir2.as_ref();
    fs::read_dir_entries(dir1, |entry_path| {
        let entry_name = entry_path.file_name().unwrap();
        if entry_path.is_dir() {
            assert_dirs_are_equal(&entry_path, &dir2.join(entry_name));
        } else {
            let path2 = dir2.join(entry_name);
            let file1 = fs::read(&entry_path);
            let file2 = fs::read(&path2);

            // We don't use `assert_eq!` because they are `Vec<u8>`, so not great for display.
            // Why not using String? Because there might be minified files or even potentially
            // binary ones, so that would display useless output.
            assert!(
                file1 == file2,
                "`{}` and `{}` have different content",
                entry_path.display(),
                path2.display(),
            );
        }
    });
}
