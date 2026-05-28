//! Test building of the `cortex-m` crate, a foundational crate in the embedded ecosystem
//! for a collection of thumb targets. This is a smoke test that verifies that both cargo
//! and rustc work in this case.
//!
//! How to run this
//! $ ./x.py clean
//! $ ./x.py test --target thumbv6m-none-eabi,thumbv7m-none-eabi tests/run-make-cargo
//!
//! Supported targets:
//! - thumbv6m-none-eabi (Bare Cortex-M0, M0+, M1)
//! - thumbv7em-none-eabi (Bare Cortex-M4, M7)
//! - thumbv7em-none-eabihf (Bare Cortex-M4F, M7F, FPU, hardfloat)
//! - thumbv7m-none-eabi (Bare Cortex-M3)

//@ only-thumb

use run_make_support::{cargo, cmd, env, env_var, target};

const CRATE: &str = "cortex-m";
const CRATE_URL: &str = "https://github.com/rust-embedded/cortex-m";
const CRATE_SHA1: &str = "a448e9156e2cb1e556e5441fd65426952ef4b927"; // v0.5.0

fn main() {
    // FIXME: requires an internet connection https://github.com/rust-lang/rust/issues/128733
    // See below link for git usage:
    // https://stackoverflow.com/questions/3489173#14091182
    cmd("git").args(["clone", CRATE_URL, CRATE]).run();
    env::set_current_dir(CRATE);
    cmd("git").args(["reset", "--hard", CRATE_SHA1]).run();

    cargo()
        .args(&[
            "build",
            "--manifest-path",
            "Cargo.toml",
            "-Zbuild-std=core",
            "--target",
            &target(),
        ])
        .env("CARGO_TARGET_DIR", "target")
        // Don't make lints fatal, but they need to at least warn
        // or they break Cargo's target info parsing.
        .env("RUSTFLAGS", "-Copt-level=0 -Cdebug-assertions=yes --cap-lints=warn")
        .run();
}
