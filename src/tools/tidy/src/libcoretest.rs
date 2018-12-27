//! Tidy check to ensure `#[test]` is not used directly inside `libcore`.
//!
//! `#![no_core]` libraries cannot be tested directly due to duplicating lang
//! item. All tests must be written externally in `libcore/tests`.

use std::path::Path;
use std::fs::read_to_string;

pub fn check(path: &Path, bad: &mut bool) {
    let libcore_path = path.join("libcore");
    super::walk(
        &libcore_path,
        &mut |subpath| t!(subpath.strip_prefix(&libcore_path)).starts_with("tests"),
        &mut |subpath| {
            if let Some("rs") = subpath.extension().and_then(|e| e.to_str()) {
                match read_to_string(subpath) {
                    Ok(contents) => {
                        if contents.contains("#[test]") {
                            tidy_error!(
                                bad,
                                "{} contains #[test]; libcore tests must be placed inside \
                                `src/libcore/tests/`",
                                subpath.display()
                            );
                        }
                    }
                    Err(err) => {
                        panic!("failed to read file {:?}: {}", subpath, err);
                    }
                }
            }
        },
    );
}
