//@ needs-target-std

#![deny(warnings)]

use run_make_support::{cargo, path, target};

fn main() {
    let target_dir = path("target");

    cargo()
        .current_dir("hello")
        .args(&[
            "build",
            "--release",
            "--manifest-path",
            "Cargo.toml",
            "-Zbuild-std",
            "--target",
            &target(),
        ])
        .env("RUSTFLAGS", "-Zunstable-options -Cpanic=immediate-abort")
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();
}
