#![feature(once_cell)]

use std::path::PathBuf;
use std::process::Command;
use test_utils::{CARGO_CLIPPY_PATH, IS_RUSTC_TEST_SUITE};

mod test_utils;

#[test]
fn test_no_deps_ignores_path_deps_in_workspaces() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = root.join("target").join("workspace_test");
    let cwd = root.join("tests/workspace_test");

    // Make sure we start with a clean state
    Command::new("cargo")
        .current_dir(&cwd)
        .env("CARGO_TARGET_DIR", &target_dir)
        .arg("clean")
        .args(&["-p", "subcrate"])
        .args(&["-p", "path_dep"])
        .output()
        .unwrap();

    // `path_dep` is a path dependency of `subcrate` that would trigger a denied lint.
    // Make sure that with the `--no-deps` argument Clippy does not run on `path_dep`.
    let output = Command::new(&*CARGO_CLIPPY_PATH)
        .current_dir(&cwd)
        .env("CARGO_INCREMENTAL", "0")
        .env("CARGO_TARGET_DIR", &target_dir)
        .arg("clippy")
        .args(&["-p", "subcrate"])
        .arg("--no-deps")
        .arg("--")
        .arg("-Cdebuginfo=0") // disable debuginfo to generate less data in the target dir
        .args(&["--cfg", r#"feature="primary_package_test""#])
        .output()
        .unwrap();
    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());

    let lint_path_dep = || {
        // Test that without the `--no-deps` argument, `path_dep` is linted.
        let output = Command::new(&*CARGO_CLIPPY_PATH)
            .current_dir(&cwd)
            .env("CARGO_INCREMENTAL", "0")
            .env("CARGO_TARGET_DIR", &target_dir)
            .arg("clippy")
            .args(&["-p", "subcrate"])
            .arg("--")
            .arg("-Cdebuginfo=0") // disable debuginfo to generate less data in the target dir
            .args(&["--cfg", r#"feature="primary_package_test""#])
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(!output.status.success());
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("error: empty `loop {}` wastes CPU cycles")
        );
    };

    // Make sure Cargo is aware of the removal of `--no-deps`.
    lint_path_dep();

    let successful_build = || {
        let output = Command::new(&*CARGO_CLIPPY_PATH)
            .current_dir(&cwd)
            .env("CARGO_INCREMENTAL", "0")
            .env("CARGO_TARGET_DIR", &target_dir)
            .arg("clippy")
            .args(&["-p", "subcrate"])
            .arg("--")
            .arg("-Cdebuginfo=0") // disable debuginfo to generate less data in the target dir
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());

        output
    };

    // Trigger a sucessful build, so Cargo would like to cache the build result.
    successful_build();

    // Make sure there's no spurious rebuild when nothing changes.
    let stderr = String::from_utf8(successful_build().stderr).unwrap();
    assert!(!stderr.contains("Compiling"));
    assert!(!stderr.contains("Checking"));
    assert!(stderr.contains("Finished"));

    // Make sure Cargo is aware of the new `--cfg` flag.
    lint_path_dep();
}
