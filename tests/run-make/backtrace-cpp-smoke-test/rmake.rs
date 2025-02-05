// Part of porting backtrace-rs tests to rustc repo:
// Original test: <https://github.com/rust-lang/rust/issues/122899>
// <https://github.com/rust-lang/backtrace-rs/tree/6fa4b85b9962c3e1be8c2e5cc605cd078134152b/crates/cpp_smoke_test>
// Issue: https://github.com/rust-lang/rust/issues/122899
// ignore-tidy-linelength

use run_make_support::{cargo, path, target};

fn main() {
    let manifest_path = path("cpp_smoke_test").join("Cargo.toml");
    let target_dir = path("cpp_smoke_test").join("target");

    cargo()
        .args(&["test", "--target", &target()])
        .args(&["--manifest-path", manifest_path.to_str().unwrap()])
        .env("CARGO_TARGET_DIR", &target_dir)
        .run();
}
