// This test ensures we are able to compile and link a simple binary with panic=immediate-abort.
// The test panic-immediate-abort-codegen checks that panic strategy produces the desired codegen,
// but is based on compiling a library crate (which is the norm for codegen tests because it is
// cleaner and more portable). So this test ensures that we didn't mix up a cfg or a compiler
// implementation detail in a way that makes panic=immediate-abort encounter errors at link time.

// Ideally this test would be run for most targets, but unfortunately:
// This test is currently written using `fn main() {}` which requires std.
// And since the default linker is only a linker for the host, we can't handle cross-compilation.
// Both of these shortcomings could be addressed at the cost of making the test more complicated.
//@ needs-target-std
//@ ignore-cross-compile

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
