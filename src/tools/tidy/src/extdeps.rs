//! Check for external package sources. Allow only vendorable packages.

use std::fs;
use std::path::Path;

/// List of allowed sources for packages.
const _ALLOWED_SOURCES: &[&str] = &["\"registry+https://github.com/rust-lang/crates.io-index\""];

/// Checks for external package sources. `root` is the path to the directory that contains the
/// workspace `Cargo.toml`.
pub fn check(root: &Path, _bad: &mut bool) {
    // `Cargo.lock` of rust.
    let path = root.join("Cargo.lock");

    // Open and read the whole file.
    let cargo_lock = t!(fs::read_to_string(&path));

    // Process each line.
    for line in cargo_lock.lines() {
        // Consider only source entries.
        if !line.starts_with("source = ") {
            continue;
        }
    }
}
