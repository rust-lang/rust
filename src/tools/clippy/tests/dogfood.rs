// Dogfood cannot run on Windows
#![cfg(not(windows))]
#![feature(once_cell)]

use std::lazy::SyncLazy;
use std::path::PathBuf;
use std::process::Command;

mod cargo;

static CLIPPY_PATH: SyncLazy<PathBuf> = SyncLazy::new(|| cargo::TARGET_LIB.join("cargo-clippy"));

#[test]
fn dogfood_clippy() {
    // run clippy on itself and fail the test if lint warnings are reported
    if cargo::is_rustc_test_suite() {
        return;
    }
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let output = Command::new(&*CLIPPY_PATH)
        .current_dir(root_dir)
        .env("CLIPPY_DOGFOOD", "1")
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy-preview")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::internal"])
        .args(&["-D", "clippy::pedantic"])
        .arg("-Cdebuginfo=0") // disable debuginfo to generate less data in the target dir
        .output()
        .unwrap();
    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}

#[test]
fn dogfood_subprojects() {
    // run clippy on remaining subprojects and fail the test if lint warnings are reported
    if cargo::is_rustc_test_suite() {
        return;
    }
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    for d in &[
        "clippy_workspace_tests",
        "clippy_workspace_tests/src",
        "clippy_workspace_tests/subcrate",
        "clippy_workspace_tests/subcrate/src",
        "clippy_dev",
        "rustc_tools_util",
    ] {
        let output = Command::new(&*CLIPPY_PATH)
            .current_dir(root_dir.join(d))
            .env("CLIPPY_DOGFOOD", "1")
            .env("CARGO_INCREMENTAL", "0")
            .arg("clippy")
            .arg("--")
            .args(&["-D", "clippy::all"])
            .args(&["-D", "clippy::pedantic"])
            .arg("-Cdebuginfo=0") // disable debuginfo to generate less data in the target dir
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());
    }
}
