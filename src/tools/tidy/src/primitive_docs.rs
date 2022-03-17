//! Tidy check to make sure `library/{std,core}/src/primitive_docs.rs` are the same file.  These are
//! different files so that relative links work properly without having to have `CARGO_PKG_NAME`
//! set, but conceptually they should always be the same.

use std::path::Path;

pub fn check(library_path: &Path, bad: &mut bool) {
    let std_name = "std/src/primitive_docs.rs";
    let core_name = "core/src/primitive_docs.rs";
    let std_contents = std::fs::read_to_string(library_path.join(std_name))
        .unwrap_or_else(|e| panic!("failed to read library/{}: {}", std_name, e));
    let core_contents = std::fs::read_to_string(library_path.join(core_name))
        .unwrap_or_else(|e| panic!("failed to read library/{}: {}", core_name, e));
    if std_contents != core_contents {
        tidy_error!(bad, "library/{} and library/{} have different contents", core_name, std_name);
    }
}
