//! This test is a part of quality control and makes clippy eat what it produces. Awesome lints and
//! long error messages
//!
//! See [Eating your own dog food](https://en.wikipedia.org/wiki/Eating_your_own_dog_food) for context

// Dogfood cannot run on Windows
#![cfg(not(windows))]
#![feature(once_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

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

    let mut command = Command::new(&*CLIPPY_PATH);
    command
        .current_dir(root_dir)
        .env("CLIPPY_DOGFOOD", "1")
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::pedantic"])
        .arg("-Cdebuginfo=0"); // disable debuginfo to generate less data in the target dir

    // internal lints only exist if we build with the internal-lints feature
    if cfg!(feature = "internal-lints") {
        command.args(&["-D", "clippy::internal"]);
    }

    let output = command.output().unwrap();

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}

fn test_no_deps_ignores_path_deps_in_workspaces() {
    if cargo::is_rustc_test_suite() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = root.join("target").join("dogfood");
    let cwd = root.join("clippy_workspace_tests");

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
    let output = Command::new(&*CLIPPY_PATH)
        .current_dir(&cwd)
        .env("CLIPPY_DOGFOOD", "1")
        .env("CARGO_INCREMENTAL", "0")
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
        let output = Command::new(&*CLIPPY_PATH)
            .current_dir(&cwd)
            .env("CLIPPY_DOGFOOD", "1")
            .env("CARGO_INCREMENTAL", "0")
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
        let output = Command::new(&*CLIPPY_PATH)
            .current_dir(&cwd)
            .env("CLIPPY_DOGFOOD", "1")
            .env("CARGO_INCREMENTAL", "0")
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

#[test]
fn dogfood_subprojects() {
    // run clippy on remaining subprojects and fail the test if lint warnings are reported
    if cargo::is_rustc_test_suite() {
        return;
    }

    // NOTE: `path_dep` crate is omitted on purpose here
    for project in &[
        "clippy_workspace_tests",
        "clippy_workspace_tests/src",
        "clippy_workspace_tests/subcrate",
        "clippy_workspace_tests/subcrate/src",
        "clippy_dev",
        "clippy_lints",
        "clippy_utils",
        "rustc_tools_util",
    ] {
        run_clippy_for_project(project);
    }

    // NOTE: Since tests run in parallel we can't run cargo commands on the same workspace at the
    // same time, so we test this immediately after the dogfood for workspaces.
    test_no_deps_ignores_path_deps_in_workspaces();
}

#[test]
#[ignore]
#[cfg(feature = "metadata-collector-lint")]
fn run_metadata_collection_lint() {
    use std::fs::File;
    use std::time::SystemTime;

    // Setup for validation
    let metadata_output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("util/gh-pages/lints.json");
    let start_time = SystemTime::now();

    // Run collection as is
    std::env::set_var("ENABLE_METADATA_COLLECTION", "1");
    run_clippy_for_project("clippy_lints");

    // Check if cargo caching got in the way
    if let Ok(file) = File::open(metadata_output_path) {
        if let Ok(metadata) = file.metadata() {
            if let Ok(last_modification) = metadata.modified() {
                if last_modification > start_time {
                    // The output file has been modified. Most likely by a hungry
                    // metadata collection monster. So We'll return.
                    return;
                }
            }
        }
    }

    // Force cargo to invalidate the caches
    filetime::set_file_mtime(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("clippy_lints/src/lib.rs"),
        filetime::FileTime::now(),
    )
    .unwrap();

    // Running the collection again
    run_clippy_for_project("clippy_lints");
}

fn run_clippy_for_project(project: &str) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut command = Command::new(&*CLIPPY_PATH);

    command
        .current_dir(root_dir.join(project))
        .env("CLIPPY_DOGFOOD", "1")
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::pedantic"])
        .arg("-Cdebuginfo=0"); // disable debuginfo to generate less data in the target dir

    // internal lints only exist if we build with the internal-lints feature
    if cfg!(feature = "internal-lints") {
        command.args(&["-D", "clippy::internal"]);
    }

    let output = command.output().unwrap();

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}
