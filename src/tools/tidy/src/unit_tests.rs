//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside `core`.
//!
//! `#![no_core]` libraries cannot be tested directly due to duplicating lang
//! items. All tests and benchmarks must be written externally in `core/{tests,benches}`.
//!
//! Outside of core tests and benchmarks should be outlined into separate files
//! named `tests.rs` or `benches.rs`, or directories named `tests` or `benches` unconfigured
//! during normal build.

use crate::walk::{filter_dirs, walk};
use std::path::Path;

pub fn check(root_path: &Path, bad: &mut bool) {
    let core = root_path.join("core");
    let core_copy = core.clone();
    let core_tests = core.join("tests");
    let core_benches = core.join("benches");
    let is_core = move |path: &Path| {
        path.starts_with(&core)
            && !(path.starts_with(&core_tests) || path.starts_with(&core_benches))
    };

    let skip = move |path: &Path, is_dir| {
        let file_name = path.file_name().unwrap_or_default();
        if is_dir {
            filter_dirs(path)
                || path.ends_with("src/doc")
                || (file_name == "tests" || file_name == "benches") && !is_core(path)
        } else {
            let extension = path.extension().unwrap_or_default();
            extension != "rs"
                || (file_name == "tests.rs" || file_name == "benches.rs") && !is_core(path)
                // UI tests with different names
                || path.ends_with("src/thread/local/dynamic_tests.rs")
                || path.ends_with("src/sync/mpsc/sync_tests.rs")
        }
    };

    walk(root_path, skip, &mut |entry, contents| {
        let path = entry.path();
        let is_core = path.starts_with(&core_copy);
        for (i, line) in contents.lines().enumerate() {
            let line = line.trim();
            let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
            let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
            if !line.starts_with("//") && (is_test() || is_bench()) {
                let explanation = if is_core {
                    "core unit tests and benchmarks must be placed into \
                         `core/tests` or `core/benches`"
                } else {
                    "unit tests and benchmarks must be placed into \
                         separate files or directories named \
                         `tests.rs`, `benches.rs`, `tests` or `benches`"
                };
                let name = if is_test() { "test" } else { "bench" };
                tidy_error!(
                    bad,
                    "`{}:{}` contains `#[{}]`; {}",
                    path.display(),
                    i + 1,
                    name,
                    explanation,
                );
                return;
            }
        }
    });
}
