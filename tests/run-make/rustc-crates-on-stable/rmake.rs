//! Checks if selected rustc crates can be compiled on the stable channel (or a "simulation" of it).
//! These crates are designed to be used by downstream users.

use run_make_support::{cargo, rustc_path, source_root};

fn main() {
    // NOTE: in the following cargo invocation, make sure that no unstable cargo flags are used! We
    // want to check that the listed compiler crates here can compile on the stable channel. We
    // can't really just "ask" a stage1 cargo to pretend that it is a stable cargo, because other
    // compiler crates are part of the same workspace, which necessarily requires that they can use
    // unstable features and experimental editions (like edition 2024).
    cargo()
        // Ensure `proc-macro2`'s nightly detection is disabled: its build script avoids using
        // nightly features when `RUSTC_STAGE` is set.
        .env("RUSTC_STAGE", "0")
        .env("RUSTC", rustc_path())
        // This forces the underlying rustc to think it is a stable rustc.
        .env("RUSTC_BOOTSTRAP", "-1")
        .arg("build")
        .arg("--manifest-path")
        .arg(source_root().join("Cargo.toml"))
        .args(&[
            // Avoid depending on transitive rustc crates
            "--no-default-features",
            // Emit artifacts in this temporary directory, not in the source_root's `target` folder
            "--target-dir",
            "target",
        ])
        // Check that these crates can be compiled on "stable"
        .args(&[
            "-p",
            "rustc_type_ir",
            "-p",
            "rustc_next_trait_solver",
            "-p",
            "rustc_pattern_analysis",
            "-p",
            "rustc_lexer",
            "-p",
            "rustc_abi",
            "-p",
            "rustc_parse_format",
        ])
        .run();
}
