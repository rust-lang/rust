//! Test building of the `cortex-m` crate, a foundational crate in the embedded ecosystem
//! for a collection of thumb targets. This is a smoke test that verifies that both cargo
//! and rustc work in this case.
//!
//! How to run this
//! $ ./x.py clean
//! $ ./x.py test --target thumbv6m-none-eabi,thumbv7m-none-eabi tests/run-make
//!
//! Supported targets:
//! - thumbv6m-none-eabi (Bare Cortex-M0, M0+, M1)
//! - thumbv7em-none-eabi (Bare Cortex-M4, M7)
//! - thumbv7em-none-eabihf (Bare Cortex-M4F, M7F, FPU, hardfloat)
//! - thumbv7m-none-eabi (Bare Cortex-M3)

//@ only-thumb

use std::path::PathBuf;

use run_make_support::rfs::create_dir;
use run_make_support::{cmd, env_var, target};

const CRATE: &str = "cortex-m";
const CRATE_URL: &str = "https://github.com/rust-embedded/cortex-m";
const CRATE_SHA1: &str = "a448e9156e2cb1e556e5441fd65426952ef4b927"; // v0.5.0

fn main() {
    // FIXME: requires an internet connection https://github.com/rust-lang/rust/issues/128733
    // See below link for git usage:
    // https://stackoverflow.com/questions/3489173#14091182
    cmd("git").args(["clone", CRATE_URL, CRATE]).run();
    std::env::set_current_dir(CRATE).unwrap();
    cmd("git").args(["reset", "--hard", CRATE_SHA1]).run();

    let target_dir = PathBuf::from("target");
    let manifest_path = PathBuf::from("Cargo.toml");

    let path = env_var("PATH");
    let rustc = env_var("RUSTC");
    let cargo = env_var("CARGO");
    // FIXME: extract cargo invocations to a proper command
    // https://github.com/rust-lang/rust/issues/128734
    let mut cmd = cmd(cargo);
    cmd.args(&[
        "build",
        "--manifest-path",
        manifest_path.to_str().unwrap(),
        "-Zbuild-std=core",
        "--target",
        &target(),
    ])
    .env("PATH", path)
    .env("RUSTC", rustc)
    .env("CARGO_TARGET_DIR", &target_dir)
    // Don't make lints fatal, but they need to at least warn
    // or they break Cargo's target info parsing.
    .env("RUSTFLAGS", "-Copt-level=0 -Cdebug-assertions=yes --cap-lints=warn");

    cmd.run();
}
