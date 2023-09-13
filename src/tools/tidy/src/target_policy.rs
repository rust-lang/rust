//! Tidy check to check that for each target in `compiler/rustc_target/src/spec/`, there is
//! documentation in `src/doc/rustc/src/platform-support/` and an assembly test in
//! `tests/assembly/targets`.

use crate::walk::filter_not_rust;
use std::path::Path;

const TARGET_DEFINITIONS_PATH: &str = "compiler/rustc_target/src/spec/";
const TARGET_DOCS_PATH: &str = "src/doc/rustc/src/platform-support/";
const TARGET_TESTS_PATH: &str = "tests/assembly/targets/";

const SKIPPED_FILES: &[&str] = &["base", "tests", "mod.rs", "abi.rs", "crt_objects.rs"];

pub fn check(root_path: &Path, bad: &mut bool) {
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
        if !test_path.exists() {
            tidy_error!(
                bad,
                "target `{}` is missing test at `{}`",
                name_no_ext,
                test_path.display()
            );
        }

        let doc_path = root_path.join(TARGET_DOCS_PATH).join(&name_no_ext).with_extension("md");
        if !doc_path.exists() {
            tidy_error!(
                bad,
                "target `{}` is missing platform support doc at `{}`",
                name_no_ext,
                doc_path.display()
            );
        }
    }
}
