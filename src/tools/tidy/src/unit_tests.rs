//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside
//! of the standard library.
//!
//! `core` and `alloc` cannot be tested directly due to duplicating lang items.
//! All tests and benchmarks must be written externally in
//! `{coretests,alloctests}/{tests,benches}`.
//!
//! Outside of the standard library, tests and benchmarks should be outlined
//! into separate files named `tests.rs` or `benches.rs`, or directories named
//! `tests` or `benches` unconfigured during normal build.

use std::path::Path;

use crate::walk::{filter_dirs, walk};

pub fn check(root_path: &Path, stdlib: bool, bad: &mut bool) {
    let skip = move |path: &Path, is_dir| {
        let file_name = path.file_name().unwrap_or_default();

        // Skip excluded directories and non-rust files
        if is_dir {
            if filter_dirs(path) || path.ends_with("src/doc") {
                return true;
            }
        } else {
            let extension = path.extension().unwrap_or_default();
            if extension != "rs" {
                return true;
            }
        }

        // Tests in a separate package are always allowed
        if is_dir && file_name != "tests" && file_name.as_encoded_bytes().ends_with(b"tests") {
            return true;
        }

        if !stdlib {
            // Outside of the standard library tests may also be in separate files in the same crate
            if is_dir {
                if file_name == "tests" || file_name == "benches" {
                    return true;
                }
            } else {
                if file_name == "tests.rs" || file_name == "benches.rs" {
                    return true;
                }
            }
        }

        if is_dir {
            // FIXME remove those exceptions once no longer necessary
            file_name == "std_detect" || file_name == "std" || file_name == "test"
        } else {
            // Tests which use non-public internals and, as such, need to
            // have the types in the same crate as the tests themselves. See
            // the comment in alloctests/lib.rs.
            path.ends_with("library/alloc/src/collections/btree/borrow/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/map/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/node/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/set/tests.rs")
                || path.ends_with("library/alloc/src/collections/linked_list/tests.rs")
                || path.ends_with("library/alloc/src/collections/vec_deque/tests.rs")
                || path.ends_with("library/alloc/src/raw_vec/tests.rs")
                || path.ends_with("library/alloc/src/wtf8/tests.rs")
        }
    };

    walk(root_path, skip, &mut |entry, contents| {
        let path = entry.path();
        let package = path
            .strip_prefix(root_path)
            .unwrap()
            .components()
            .next()
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap();
        for (i, line) in contents.lines().enumerate() {
            let line = line.trim();
            let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
            let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
            if !line.starts_with("//") && (is_test() || is_bench()) {
                let explanation = if stdlib {
                    format!(
                        "`{package}` unit tests and benchmarks must be placed into `{package}tests`"
                    )
                } else {
                    "unit tests and benchmarks must be placed into \
                         separate files or directories named \
                         `tests.rs`, `benches.rs`, `tests` or `benches`"
                        .to_owned()
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
