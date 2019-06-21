//! Tidy check to ensure `#[test]` is not used directly inside `libcore`.
//!
//! `#![no_core]` libraries cannot be tested directly due to duplicating lang
//! item. All tests must be written externally in `libcore/tests`.

use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    let libcore_path = path.join("libcore");
    super::walk(
        &libcore_path,
        &mut |subpath| t!(subpath.strip_prefix(&libcore_path)).starts_with("tests"),
        &mut |entry, contents| {
            let subpath = entry.path();
            if let Some("rs") = subpath.extension().and_then(|e| e.to_str()) {
                if contents.contains("#[test]") {
                    tidy_error!(
                        bad,
                        "{} contains #[test]; libcore tests must be placed inside \
                        `src/libcore/tests/`",
                        subpath.display()
                    );
                }
            }
        },
    );
}
