//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside `libcore`.
//!
//! `#![no_core]` libraries cannot be tested directly due to duplicating lang
//! items. All tests and benchmarks must be written externally in `libcore/{tests,benches}`.
//!
//! Outside of libcore tests and benchmarks should be outlined into separate files
//! named `tests.rs` or `benches.rs`, or directories named `tests` or `benches` unconfigured
//! during normal build.

use std::path::Path;

pub fn check(root_path: &Path, bad: &mut bool) {
    let libcore = &root_path.join("libcore");
    let libcore_tests = &root_path.join("libcore/tests");
    let libcore_benches = &root_path.join("libcore/benches");
    let is_core = |path: &Path| {
        path.starts_with(libcore) &&
        !(path.starts_with(libcore_tests) || path.starts_with(libcore_benches))
    };

    let mut skip = |path: &Path| {
        let file_name = path.file_name().unwrap_or_default();
        if path.is_dir() {
            super::filter_dirs(path) ||
            path.ends_with("src/test") ||
            path.ends_with("src/doc") ||
            path.ends_with("src/libstd") || // FIXME?
            (file_name == "tests" || file_name == "benches") && !is_core(path)
        } else {
            let extension = path.extension().unwrap_or_default();
            extension != "rs" ||
            (file_name == "tests.rs" || file_name == "benches.rs") && !is_core(path)
        }
    };

    super::walk(
        root_path,
        &mut skip,
        &mut |entry, contents| {
            let path = entry.path();
            let is_libcore = path.starts_with(libcore);
            for (i, line) in contents.lines().enumerate() {
                let line = line.trim();
                let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
                let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
                if !line.starts_with("//") && (is_test() || is_bench()) {
                    let explanation = if is_libcore {
                        "libcore unit tests and benchmarks must be placed into \
                         `libcore/tests` or `libcore/benches`"
                    } else {
                        "unit tests and benchmarks must be placed into \
                         separate files or directories named \
                         `tests.rs`, `benches.rs`, `tests` or `benches`"
                    };
                    let name = if is_test() { "test" } else { "bench" };
                    tidy_error!(
                        bad, "`{}:{}` contains `#[{}]`; {}",
                        path.display(), i + 1, name, explanation,
                    );
                    return;
                }
            }
        },
    );
}
