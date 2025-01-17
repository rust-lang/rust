//! This test runs a basic application for thumb targets, using the cortex-m crate.
//!
//! These targets are very bare-metal: the first instruction the core runs on
//! power-on is already user code. The cortex-m-rt has to initialize the stack, .data,
//! .bss, enable the FPU if present, etc.
//!
//! This test builds and runs the applications for various thumb targets using qemu.
//!
//! How to run this
//! $ ./x.py clean
//! $ ./x.py test --target thumbv6m-none-eabi,thumbv7m-none-eabi tests/run-make
//!
//! For supported targets, see `example/.cargo/config.toml`
//!
//! FIXME: https://github.com/rust-lang/rust/issues/128733 this test uses external
//! dependencies, and needs an active internet connection

//@ only-thumb

use std::path::PathBuf;

use run_make_support::{cargo, cmd, env_var, path, target};

const CRATE: &str = "example";

fn main() {
    std::env::set_current_dir(CRATE).unwrap();

    let target_dir = path("target");

    // Debug
    cargo()
        .args(&["run", "--target", &target()])
        .env("RUSTFLAGS", "-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x")
        .env("CARGO_TARGET_DIR", &target_dir)
        .run()
        .assert_stdout_contains("x = 42");

    // Release
    cargo()
        .args(&["run", "--release", "--target", &target()])
        .env("RUSTFLAGS", "-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x")
        .env("CARGO_TARGET_DIR", &target_dir)
        .run()
        .assert_stdout_contains("x = 42");
}
