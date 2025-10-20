//! Check for external package sources. Allow only vendorable packages.

use std::fs;
use std::io::{BufRead as _, BufReader};
use std::path::Path;

use crate::deps::WorkspaceInfo;
use crate::diagnostics::DiagCtx;

/// List of allowed sources for packages.
const ALLOWED_SOURCES: &[&str] = &[
    "registry+https://github.com/rust-lang/crates.io-index",
    // This is `rust_team_data` used by `site` in src/tools/rustc-perf,
    "git+https://github.com/rust-lang/team#a5260e76d3aa894c64c56e6ddc8545b9a98043ec",
];

/// Checks for external package sources. `root` is the path to the directory that contains the
/// workspace `Cargo.toml`.
pub fn check(root: &Path, diag_ctx: DiagCtx) {
    let mut check = diag_ctx.start_check("extdeps");

    for &WorkspaceInfo { path, submodules, .. } in crate::deps::WORKSPACES {
        if crate::deps::has_missing_submodule(root, submodules) {
            continue;
        }

        let lockfile = root.join(path).join("Cargo.lock");

        if !lockfile.exists() {
            check.error(format!("the `{path}` workspace doesn't have a Cargo.lock"));
            continue;
        }

        // At the time of writing, longest line in lockfile is 82 bytes.
        let cargo_lock = BufReader::with_capacity(128, t!(fs::File::open(&lockfile)));
        let mut lines = cargo_lock.lines();

        // Process each line.
        while let Some(line) = t!(lines.next().transpose()) {
            if let Some(source) = line.strip_prefix(r#"source = ""#)
                && let Some(source) = source.strip_suffix(r#"""#)
                && !ALLOWED_SOURCES.contains(&source)
            {
                check.error(format!("invalid source: {}", source));
            }
        }
    }
}
