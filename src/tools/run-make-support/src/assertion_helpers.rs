//! Collection of assertions and assertion-related helpers.

use std::panic;
use std::path::Path;

use crate::{fs, regex};

fn print<'a, 'e, A: AsRef<str>, E: AsRef<str>>(
    assertion_kind: &str,
    haystack: &'a A,
    needle: &'e E,
) -> (&'a str, &'e str) {
    let haystack = haystack.as_ref();
    let needle = needle.as_ref();
    eprintln!("{assertion_kind}:");
    eprintln!("=== HAYSTACK ===");
    eprintln!("{}", haystack);
    eprintln!("=== NEEDLE ===");
    eprintln!("{}", needle);
    (haystack, needle)
}

/// Assert that `actual` is equal to `expected`.
#[track_caller]
pub fn assert_equals<A: AsRef<str>, E: AsRef<str>>(actual: A, expected: E) {
    let actual = actual.as_ref();
    let expected = expected.as_ref();
    eprintln!("=== ACTUAL TEXT ===");
    eprintln!("{}", actual);
    eprintln!("=== EXPECTED ===");
    eprintln!("{}", expected);
    if actual != expected {
        panic!("expected text was not found in actual text");
    }
}

/// Assert that `haystack` contains `needle`.
#[track_caller]
pub fn assert_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let (haystack, needle) = print("assert_contains", &haystack, &needle);
    if !haystack.contains(needle) {
        panic!("needle was not found in haystack");
    }
}

/// Assert that `haystack` does not contain `needle`.
#[track_caller]
pub fn assert_not_contains<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let (haystack, needle) = print("assert_not_contains", &haystack, &needle);
    if haystack.contains(needle) {
        panic!("needle was unexpectedly found in haystack");
    }
}

/// Assert that `haystack` contains the regex pattern `needle`.
#[track_caller]
pub fn assert_contains_regex<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let (haystack, needle) = print("assert_contains_regex", &haystack, &needle);
    let re = regex::Regex::new(needle).unwrap();
    if !re.is_match(haystack) {
        panic!("needle was not found in haystack");
    }
}

/// Assert that `haystack` does not contain the regex pattern `needle`.
#[track_caller]
pub fn assert_not_contains_regex<H: AsRef<str>, N: AsRef<str>>(haystack: H, needle: N) {
    let (haystack, needle) = print("assert_not_contains_regex", &haystack, &needle);
    let re = regex::Regex::new(needle).unwrap();
    if re.is_match(haystack) {
        panic!("needle was unexpectedly found in haystack");
    }
}

/// Assert that `haystack` contains `needle` a `count` number of times.
#[track_caller]
pub fn assert_count_is<H: AsRef<str>, N: AsRef<str>>(count: usize, haystack: H, needle: N) {
    let (haystack, needle) = print("assert_count_is", &haystack, &needle);
    if count != haystack.matches(needle).count() {
        panic!("needle did not appear {count} times in haystack");
    }
}

/// Assert that all files in `dir1` exist and have the same content in `dir2`
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
