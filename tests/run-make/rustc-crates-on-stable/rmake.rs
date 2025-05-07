//! Checks if selected rustc crates can be compiled on the stable channel (or a "simulation" of it).
//! These crates are designed to be used by downstream users.

use run_make_support::{cargo, rustc_path, source_root};

fn main() {
    // Use the stage0 beta cargo for the compilation (it shouldn't really matter which cargo we use)
    cargo()
        // Ensure `proc-macro2`'s nightly detection is disabled
        .env("RUSTC_STAGE", "0")
        .env("RUSTC", rustc_path())
        // We want to disallow all nightly features to simulate a stable build
        .env("RUSTFLAGS", "-Zallow-features=")
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
            "-p",
            "rustc_hashes",
        ])
        .run();
}
