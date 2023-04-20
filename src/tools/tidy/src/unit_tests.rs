//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside `core`.
//!
//! `#![no_core]` libraries cannot be tested directly due to duplicating lang
//! items. All tests and benchmarks must be written externally in `core/{tests,benches}`.

use crate::walk::{filter_dirs, walk};
use std::path::Path;

pub fn check(root_path: &Path, bad: &mut bool) {
    let core = root_path.join("core");
    let core_tests = core.join("tests");
    let core_benches = core.join("benches");
    let is_core = move |path: &Path| {
        path.starts_with(&core)
            && !(path.starts_with(&core_tests) || path.starts_with(&core_benches))
    };

    let skip = move |path: &Path, is_dir| {
        if is_dir {
            filter_dirs(path) || !is_core(path)
        } else {
            let extension = path.extension().unwrap_or_default();
            extension != "rs" || !is_core(path)
        }
    };

    walk(root_path, skip, &mut |entry, contents| {
        let path = entry.path();
        for (i, line) in contents.lines().enumerate() {
            let line = line.trim();
            let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
            let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
            if !line.starts_with("//") && (is_test() || is_bench()) {
                let name = if is_test() { "test" } else { "bench" };
                tidy_error!(
                    bad,
                    "`{}:{}` contains `#[{}]`;\
                     core unit tests and benchmarks must be placed into `core/tests` or `core/benches`",
                    path.display(),
                    i + 1,
                    name,
                );
                return;
            }
        }
    });
}
