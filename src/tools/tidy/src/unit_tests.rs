//! Tidy check to ensure `#[test]` and `#[bench]` are not used directly inside
//! `core` or `alloc`.
//!
//! `core` and `alloc` cannot be tested directly due to duplicating lang items.
//! All tests and benchmarks must be written externally in
//! `{coretests,alloctests}/{tests,benches}`.
//!
//! Outside of `core` and `alloc`, tests and benchmarks should be outlined into
//! separate files named `tests.rs` or `benches.rs`, or directories named
//! `tests` or `benches` unconfigured during normal build.

use std::path::Path;

use crate::walk::{filter_dirs, walk};

pub fn check(root_path: &Path, bad: &mut bool) {
    let core = root_path.join("core");
    let core_copy = core.clone();
    let is_core = move |path: &Path| path.starts_with(&core);
    let alloc = root_path.join("alloc");
    let alloc_copy = alloc.clone();
    let is_alloc = move |path: &Path| path.starts_with(&alloc);

    let skip = move |path: &Path, is_dir| {
        let file_name = path.file_name().unwrap_or_default();
        if is_dir {
            filter_dirs(path)
                || path.ends_with("src/doc")
                || (file_name == "tests" || file_name == "benches")
                    && !is_core(path)
                    && !is_alloc(path)
        } else {
            let extension = path.extension().unwrap_or_default();
            extension != "rs"
                || (file_name == "tests.rs" || file_name == "benches.rs")
                    && !is_core(path)
                    && !is_alloc(path)
                // Tests which use non-public internals and, as such, need to
                // have the types in the same crate as the tests themselves. See
                // the comment in alloctests/lib.rs.
                || path.ends_with("library/alloc/src/collections/btree/borrow/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/map/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/node/tests.rs")
                || path.ends_with("library/alloc/src/collections/btree/set/tests.rs")
                || path.ends_with("library/alloc/src/collections/linked_list/tests.rs")
                || path.ends_with("library/alloc/src/collections/vec_deque/tests.rs")
                || path.ends_with("library/alloc/src/raw_vec/tests.rs")
        }
    };

    walk(root_path, skip, &mut |entry, contents| {
        let path = entry.path();
        let is_core = path.starts_with(&core_copy);
        let is_alloc = path.starts_with(&alloc_copy);
        for (i, line) in contents.lines().enumerate() {
            let line = line.trim();
            let is_test = || line.contains("#[test]") && !line.contains("`#[test]");
            let is_bench = || line.contains("#[bench]") && !line.contains("`#[bench]");
            if !line.starts_with("//") && (is_test() || is_bench()) {
                let explanation = if is_core {
                    "`core` unit tests and benchmarks must be placed into `coretests`"
                } else if is_alloc {
                    "`alloc` unit tests and benchmarks must be placed into `alloctests`"
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
