use std::env;
use std::path::PathBuf;

use super::rustc_info::get_file_name;
use super::utils::{cargo_command, is_ci};

pub(crate) fn build_backend(
    channel: &str,
    host_triple: &str,
    use_unstable_features: bool,
) -> PathBuf {
    let source_dir = std::env::current_dir().unwrap();
    let mut cmd = cargo_command("cargo", "build", Some(host_triple), &source_dir);

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

    source_dir
        .join("target")
        .join(host_triple)
        .join(channel)
        .join(get_file_name("rustc_codegen_cranelift", "dylib"))
}
