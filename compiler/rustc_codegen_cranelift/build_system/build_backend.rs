use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::utils::is_ci;

pub(crate) fn build_backend(
    channel: &str,
    host_triple: &str,
    use_unstable_features: bool,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.arg("build").arg("--target").arg(host_triple);

    cmd.env("CARGO_BUILD_INCREMENTAL", "true"); // Force incr comp even in release mode

    let mut rustflags = env::var("RUSTFLAGS").unwrap_or_default();

    if is_ci() {
        // Deny warnings on CI
        rustflags += " -Dwarnings";

        // Disabling incr comp reduces cache size and incr comp doesn't save as much on CI anyway
        cmd.env("CARGO_BUILD_INCREMENTAL", "false");
    }

    if use_unstable_features {
        cmd.arg("--features").arg("unstable-features");
    }

    match channel {
        "debug" => {}
        "release" => {
            cmd.arg("--release");
        }
        _ => unreachable!(),
    }

    cmd.env("RUSTFLAGS", rustflags);

    eprintln!("[BUILD] rustc_codegen_cranelift");
    super::utils::spawn_and_wait(cmd);

    Path::new("target").join(host_triple).join(channel)
}
