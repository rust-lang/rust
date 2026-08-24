//! This test checks that Clippy files (source and tests):
//!
//! - only have one top-level lint attribute of each kind (`allow`/`deny`/`expect`/`forbid`/`warn`)
//! - put unqualified lint names before the qualified ones
//! - use alphabetical order inside each qualification
//!
//! This test is disabled when ran as part of the compiler test suite, to prevent
//! changes from being difficult to make there. Incorrect organizations will be
//! fixed during the merge process.
//!
//! This test may eventually be replaced by a restriction lint which imposes
//! constraints on attributes. Right now, this regex-based one does the job
//! on Clippy test files. It only checks top-level beginning-of-the-line attributes,
//! so that it won't trigger on a `#![cfg_attr(…, warn(…))]` which can legitimately be
//! used in addition to another `#![warn(…)]` attribute.

use regex::Regex;
use std::collections::HashSet;
use std::{fs, iter};
use test_utils::IS_RUSTC_TEST_SUITE;
use walkdir::{DirEntry, WalkDir};

mod test_utils;

const SKIPPED_FILES: [&str; 7] = [
    "./tests/lint-attributes-organization.rs", // this file, for the sanity checks
    "./tests/ui/blanket_clippy_restriction_lints.rs", // separate lines are better
    "./tests/ui/deprecated.rs",                // generated
    "./tests/ui/duplicated_attributes.rs",     // obviously
    "./tests/ui/rename.rs",                    // generated
    "./tests/ui/unknown_clippy_lints.rs",      // separate lines are better
    "./target/",                               // generated files
];

#[test]
fn lint_attribute_organization() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }
    let attribute_regex = attribute_regex();
    let mut problem_found = false;
    for path in WalkDir::new(".")
        .into_iter()
        .flatten()
        .map(DirEntry::into_path)
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "rs") && !SKIPPED_FILES.iter().any(|&skipped| p.starts_with(skipped))
        })
    {
        for diag in error_for_content(&fs::read_to_string(&path).unwrap(), &attribute_regex) {
            eprintln!("ERROR: {}: {diag}", path.display());
            problem_found = true;
        }
    }
    assert!(
        !problem_found,
        "some lint attributes do not meet the attributes organization requirements"
    );
}

fn attribute_regex() -> Regex {
    Regex::new(r"(?ms)^#!\[(allow|deny|expect|forbid|warn)\((.+?)\)\]").unwrap()
}

// Check if file content contains only ordered lint names in top-level inner attributes, and
// also that every attribute is present only once.
fn error_for_content(content: &str, attribute_regex: &Regex) -> Vec<String> {
    let mut diags = vec![];
    let mut attributes = HashSet::new();
    for cap in attribute_regex.captures_iter(content) {
        let attribute = &cap[1];
        if !attributes.insert(attribute.to_owned()) {
            diags.push(format!("duplicate top-level `#![{attribute}]` attribute"));
        }
        let lint_names = cap[2]
            .lines()
            // Split off any trailing comments
            .map(|l| l.split_once("//").map_or(l, |(b, _)| b).trim())
            .flat_map(|l| l.split_whitespace().map(|s| s.strip_suffix(',').unwrap_or(s)))
            .collect::<Vec<_>>();
        for [a, b] in lint_names.array_windows() {
            if a.contains("::") && !b.contains("::") {
                diags.push(format!(
                    "qualified lint names (`{a}`) must come after unqualified ones (`{b}`)"
                ));
            } else if !a.contains("::") && b.contains("::") {
            } else if a > b {
                diags.push(format!("lint names must be ordered: `{b}` must come before `{a}`"));
            }
        }
    }
    diags
}

#[test]
fn internal_sanity_check() {
    let attribute_regex = attribute_regex();
    check_errors(
        "
#![allow(
  clippy::def, // Comment
  clippy::abc
)]
#![allow(clippy::zyx, unknown)]",
        vec![
            "lint names must be ordered: `clippy::abc` must come before `clippy::def`",
            "duplicate top-level `#![allow]` attribute",
            "qualified lint names (`clippy::zyx`) must come after unqualified ones (`unknown`)",
        ],
        &attribute_regex,
    );
    check_errors(
        "#![warn(before, after)]",
        vec!["lint names must be ordered: `after` must come before `before`"],
        &attribute_regex,
    );
    check_errors(
        "#![deny(clippy::abc, clippy::def, clippy::zyx)]",
        vec![],
        &attribute_regex,
    );
}

fn check_errors(content: &str, expected: Vec<&str>, attribute_regex: &Regex) {
    let errors = error_for_content(content, attribute_regex);
    assert_eq!(errors.len(), expected.len());
    for (i, (error, expected)) in iter::zip(errors, expected).enumerate() {
        assert_eq!(&error, expected, "the {}-th error differs from the expectation", i + 1);
    }
}
