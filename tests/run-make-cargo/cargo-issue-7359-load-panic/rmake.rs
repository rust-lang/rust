// This is a regression test to ensure that rustc doesn't load `panic_abort`
// from the sysroot. See rust-lang/cargo#7359
//
//@ needs-target-std

use run_make_support::{cargo, path, target};

fn main() {
    let target_dir = path("target");

    cargo()
        .args(&[
            "build",
            "--release",
            "--manifest-path",
            "Cargo.toml",
            "-Zbuild-std=std",
            "--target",
            &target(),
        ])
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();
}
