//! Check for external package sources. Allow only vendorable packages.

use build_helper::ci::CiEnv;
use std::fs::{self, read_dir};
use std::path::Path;

/// List of allowed sources for packages.
const ALLOWED_SOURCES: &[&str] = &[
    r#""registry+https://github.com/rust-lang/crates.io-index""#,
    // This is `rust_team_data` used by `site` in src/tools/rustc-perf,
    r#""git+https://github.com/rust-lang/team#a5260e76d3aa894c64c56e6ddc8545b9a98043ec""#,
];

/// Checks for external package sources. `root` is the path to the directory that contains the
/// workspace `Cargo.toml`.
pub fn check(root: &Path, bad: &mut bool) {
    let submodules = build_helper::util::parse_gitmodules(root);
    for &(workspace, _, _) in crate::deps::WORKSPACES {
        // Skip if it's a submodule, not in a CI environment, and not initialized.
        //
        // This prevents enforcing developers to fetch submodules for tidy.
        if submodules.contains(&workspace.into())
            && !CiEnv::is_ci()
            // If the directory is empty, we can consider it as an uninitialized submodule.
            && read_dir(root.join(workspace)).unwrap().next().is_none()
        {
            continue;
        }

        // FIXME check other workspaces too
        // `Cargo.lock` of rust.
        let path = root.join(workspace).join("Cargo.lock");

        if !path.exists() {
            tidy_error!(bad, "the `{workspace}` workspace doesn't have a Cargo.lock");
            continue;
        }

        // Open and read the whole file.
        let cargo_lock = t!(fs::read_to_string(&path));

        // Process each line.
        for line in cargo_lock.lines() {
            // Consider only source entries.
            if !line.starts_with("source = ") {
                continue;
            }

            // Extract source value.
            let source = line.split_once('=').unwrap().1.trim();

            // Ensure source is allowed.
            if !ALLOWED_SOURCES.contains(&&*source) {
                tidy_error!(bad, "invalid source: {}", source);
            }
        }
    }
}
