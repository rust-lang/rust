//! Checks if selected rustc crates can be compiled on the stable channel (or a "simulation" of it).
//! These crates are designed to be used by downstream users.

use run_make_support::{cargo, rustc_path, source_root};

fn main() {
    // Use the stage0 beta cargo for the compilation (it shouldn't really matter which cargo we use)
    let cargo = cargo()
        // This is required to allow using nightly cargo features (public-dependency) with beta
        // cargo
        .env("RUSTC_BOOTSTRAP", "1")
        .env("RUSTC_STAGE", "0") // Ensure `proc-macro2`'s nightly detection is disabled
        .env("RUSTC", rustc_path())
        .arg("build")
        .arg("--manifest-path")
        .arg(source_root().join("Cargo.toml"))
        .args(&[
            "--config",
            r#"workspace.exclude=["library/core"]"#,
            // We want to disallow all nightly features, to simulate a stable build
            // public-dependency needs to be enabled for cargo to work
            "-Zallow-features=public-dependency",
            // Avoid depending on transitive rustc crates
            "--no-default-features",
            // Check that these crates can be compiled on "stable"
            "-p",
            "rustc_type_ir",
            "-p",
            "rustc_next_trait_solver",
            "-p",
            "rustc_pattern_analysis",
            "-p",
            "rustc_lexer",
        ])
        .run();
}
