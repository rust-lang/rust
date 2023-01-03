//! To run this test, use
//! `env INTEGRATION=rust-lang/log cargo test --test integration --features=integration`
//!
//! You can use a different `INTEGRATION` value to test different repositories.

#![cfg(feature = "integration")]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, eprintln};

#[cfg(not(windows))]
const CARGO_CLIPPY: &str = "cargo-clippy";
#[cfg(windows)]
const CARGO_CLIPPY: &str = "cargo-clippy.exe";

// NOTE: arguments passed to the returned command will be `clippy-driver` args, not `cargo-clippy`
// args. Use `cargo_args` to pass arguments to cargo-clippy.
fn clippy_command(repo_dir: &Path, cargo_args: &[&str]) -> Command {
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = option_env!("CARGO_TARGET_DIR").map_or_else(|| root_dir.join("target"), PathBuf::from);
    let clippy_binary = target_dir.join(env!("PROFILE")).join(CARGO_CLIPPY);

    let mut cargo_clippy = Command::new(clippy_binary);
    cargo_clippy
        .current_dir(repo_dir)
        .env("RUST_BACKTRACE", "full")
        .env("CARGO_TARGET_DIR", root_dir.join("target"))
        .args(["clippy", "--all-targets", "--all-features"])
        .args(cargo_args)
        .args(["--", "--cap-lints", "warn", "-Wclippy::pedantic", "-Wclippy::nursery"]);
    cargo_clippy
}

/// Return a directory with a checkout of the repository in `INTEGRATION`.
fn repo_dir(repo_name: &str) -> PathBuf {
    let repo_url = format!("https://github.com/{repo_name}");
    let crate_name = repo_name
        .split('/')
        .nth(1)
        .expect("repo name should have format `<org>/<name>`");

    let mut repo_dir = tempfile::tempdir().expect("couldn't create temp dir").into_path();
    repo_dir.push(crate_name);

    let st = Command::new("git")
        .args([
            OsStr::new("clone"),
            OsStr::new("--depth=1"),
            OsStr::new(&repo_url),
            OsStr::new(&repo_dir),
        ])
        .status()
        .expect("unable to run git");
    assert!(st.success());

    repo_dir
}

#[cfg_attr(feature = "integration", test)]
fn integration_test() {
    let repo_name = env::var("INTEGRATION").expect("`INTEGRATION` var not set");
    let repo_dir = repo_dir(&repo_name);
    let output = clippy_command(&repo_dir, &[]).output().expect("failed to run clippy");
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprintln!("{stderr}");
    }

    if let Some(backtrace_start) = stderr.find("error: internal compiler error") {
        static BACKTRACE_END_MSG: &str = "end of query stack";
        let backtrace_end = stderr[backtrace_start..]
            .find(BACKTRACE_END_MSG)
            .expect("end of backtrace not found");

        panic!(
            "internal compiler error\nBacktrace:\n\n{}",
            &stderr[backtrace_start..backtrace_start + backtrace_end + BACKTRACE_END_MSG.len()]
        );
    } else if stderr.contains("query stack during panic") {
        panic!("query stack during panic in the output");
    } else if stderr.contains("E0463") {
        // Encountering E0463 (can't find crate for `x`) did _not_ cause the build to fail in the
        // past. Even though it should have. That's why we explicitly panic here.
        // See PR #3552 and issue #3523 for more background.
        panic!("error: E0463");
    } else if stderr.contains("E0514") {
        panic!("incompatible crate versions");
    } else if stderr.contains("failed to run `rustc` to learn about target-specific information") {
        panic!("couldn't find librustc_driver, consider setting `LD_LIBRARY_PATH`");
    } else {
        assert!(
            !stderr.contains("toolchain") || !stderr.contains("is not installed"),
            "missing required toolchain"
        );
    }

    match output.status.code() {
        Some(0) => println!("Compilation successful"),
        Some(code) => eprintln!("Compilation failed. Exit code: {code}"),
        None => panic!("Process terminated by signal"),
    }
}

#[cfg_attr(feature = "integration", test)]
fn test_sysroot() {
    #[track_caller]
    fn verify_cmd(cmd: &mut Command) {
        // Test that SYSROOT is ignored if `--sysroot` is passed explicitly.
        cmd.env("SYSROOT", "/dummy/value/does/not/exist");
        // We don't actually care about emitting lints, we only want to verify clippy doesn't give a hard
        // error.
        cmd.arg("-Awarnings");
        let output = cmd.output().expect("failed to run clippy");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.is_empty(), "clippy printed an error: {stderr}");
        assert!(output.status.success(), "clippy exited with an error");
    }

    let rustc = std::env::var("RUSTC").unwrap_or("rustc".to_string());
    let rustc_output = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("unable to run rustc");
    assert!(rustc_output.status.success());
    let sysroot = String::from_utf8(rustc_output.stdout).unwrap();
    let sysroot = sysroot.trim_end();

    // This is a fairly small repo; we want to avoid checking out anything heavy twice, so just
    // hard-code it.
    let repo_name = "rust-lang/log";
    let repo_dir = repo_dir(repo_name);
    // Pass the sysroot through RUSTFLAGS.
    verify_cmd(clippy_command(&repo_dir, &["--quiet"]).env("RUSTFLAGS", format!("--sysroot={sysroot}")));
    // NOTE: we don't test passing the arguments directly to clippy-driver (with `-- --sysroot`)
    // because it breaks for some reason. I (@jyn514) haven't taken time to track down the bug
    // because rust-lang/rust uses RUSTFLAGS and nearly no one else uses --sysroot.
}
