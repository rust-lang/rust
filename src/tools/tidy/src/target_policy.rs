//! Tidy check to check that for each target in `compiler/rustc_target/src/spec/`, there is
//! documentation in `src/doc/rustc/src/platform-support/` and an assembly test in
//! `tests/assembly/targets`.

use crate::walk::{filter_not_markdown, filter_not_rust};
use std::{collections::HashSet, path::Path};

const TARGET_DEFINITIONS_PATH: &str = "compiler/rustc_target/src/spec/";
const TARGET_DOCS_PATH: &str = "src/doc/rustc/src/platform-support/";
const TARGET_TESTS_PATH: &str = "tests/assembly/targets/";

const SKIPPED_FILES: &[&str] =
    &["base", "tests", "mod.rs", "abi.rs", "crt_objects.rs", "TEMPLATE.md"];

pub fn check(root_path: &Path, bad: &mut bool) {
    let mut expected_docs = HashSet::new();
    let mut expected_tests = HashSet::new();

    let definitions_path = root_path.join(TARGET_DEFINITIONS_PATH);
    for defn in ignore::WalkBuilder::new(&definitions_path)
        .max_depth(Some(1))
        .filter_entry(|e| {
            !filter_not_rust(e.path()) && !SKIPPED_FILES.contains(&e.file_name().to_str().unwrap())
        })
        .build()
    {
        let defn = defn.unwrap();
        // `WalkBuilder` always visits the directory at the root of the walk first, regardless
        // of the filters.
        if defn.path() == definitions_path {
            continue;
        }

        let path = defn.path();
        let name = path.file_name().unwrap().to_str().unwrap();
        let name_no_ext = path.file_stem().unwrap().to_str().unwrap();

        let test_path = root_path.join(TARGET_TESTS_PATH).join(&name);
        expected_tests.insert(test_path.clone());
        if !test_path.exists() {
            tidy_error!(
                bad,
                "target `{}` is missing test at `{}`",
                name_no_ext,
                test_path.display()
            );
        }

        let doc_path = root_path.join(TARGET_DOCS_PATH).join(&name_no_ext).with_extension("md");
        expected_docs.insert(doc_path.clone());
        if !doc_path.exists() {
            tidy_error!(
                bad,
                "target `{}` is missing platform support doc at `{}`",
                name_no_ext,
                doc_path.display()
            );
        }
    }

    let tests_path = root_path.join(TARGET_TESTS_PATH);
    for defn in ignore::WalkBuilder::new(&tests_path)
        .max_depth(Some(1))
        .filter_entry(|e| {
            !filter_not_rust(e.path()) && !SKIPPED_FILES.contains(&e.file_name().to_str().unwrap())
        })
        .build()
    {
        let defn = defn.unwrap();
        let path = defn.path();
        if path == tests_path {
            continue;
        }

        if !expected_tests.contains(path) {
            tidy_error!(bad, "target test `{}` exists but target does not", path.display());
        }
    }

    let docs_path = root_path.join(TARGET_DOCS_PATH);
    for defn in ignore::WalkBuilder::new(&docs_path)
        .max_depth(Some(1))
        .filter_entry(|e| {
            !filter_not_markdown(e.path())
                && !SKIPPED_FILES.contains(&e.file_name().to_str().unwrap())
        })
        .build()
    {
        let defn = defn.unwrap();
        let path = defn.path();
        if path == docs_path {
            continue;
        }

        if !expected_docs.contains(path) {
            tidy_error!(
                bad,
                "platform support doc `{}` exists but target does not",
                path.display()
            );
        }
    }
}
