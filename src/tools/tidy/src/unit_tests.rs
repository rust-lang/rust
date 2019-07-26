//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside
//! `libcore` or `liballoc`.
//!
//! `#![no_std]` libraries cannot be tested directly due to duplicating lang
//! items. All tests and benchmarks must be written externally in `libcore/{tests,benches}`
//! or `liballoc/{tests,benches}`.
//!
//! Outside of libcore and liballoc tests and benchmarks should be outlined into separate files
//! named `tests.rs` or `benches.rs`, or directories named `tests` or `benches` unconfigured
//! during normal build.

use std::path::Path;

pub fn check(root_path: &Path, bad: &mut bool) {
    let libcore = &root_path.join("libcore");
    let liballoc = &root_path.join("liballoc");
    let libcore_tests = &root_path.join("libcore/tests");
    let liballoc_tests = &root_path.join("liballoc/tests");
    let libcore_benches = &root_path.join("libcore/benches");
    let liballoc_benches = &root_path.join("liballoc/benches");
    let is_core_or_alloc = |path: &Path| {
        let is_core = path.starts_with(libcore) &&
                      !(path.starts_with(libcore_tests) || path.starts_with(libcore_benches));
        let is_alloc = path.starts_with(liballoc) &&
                       !(path.starts_with(liballoc_tests) || path.starts_with(liballoc_benches));
        is_core || is_alloc
    };
    let fixme = [
        "liballoc",
        "libpanic_unwind/dwarf",
        "librustc",
        "librustc_data_structures",
        "librustc_incremental/persist",
        "librustc_lexer/src",
        "librustc_target/spec",
        "librustdoc",
        "libserialize",
        "libstd",
        "libsyntax",
        "libsyntax_pos",
        "libterm/terminfo",
        "libtest",
        "tools/compiletest/src",
        "tools/tidy/src",
    ];

    let mut skip = |path: &Path| {
        let file_name = path.file_name().unwrap_or_default();
        if path.is_dir() {
            super::filter_dirs(path) ||
            path.ends_with("src/test") ||
            path.ends_with("src/doc") ||
            (file_name == "tests" || file_name == "benches") && !is_core_or_alloc(path) ||
            fixme.iter().any(|p| path.ends_with(p))
        } else {
            let extension = path.extension().unwrap_or_default();
            extension != "rs" ||
            (file_name == "tests.rs" || file_name == "benches.rs") && !is_core_or_alloc(path)
        }
    };

    super::walk(
        root_path,
        &mut skip,
        &mut |entry, contents| {
            let path = entry.path();
            let is_libcore = path.starts_with(libcore);
            let is_liballoc = path.starts_with(liballoc);
            for (i, line) in contents.lines().enumerate() {
                let line = line.trim();
                let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
                let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
                if !line.starts_with("//") && (is_test() || is_bench()) {
                    let explanation = if is_libcore {
                        "libcore unit tests and benchmarks must be placed into \
                         `libcore/tests` or `libcore/benches`"
                    } else if is_liballoc {
                        "liballoc unit tests and benchmarks must be placed into \
                         `liballoc/tests` or `liballoc/benches`"
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
