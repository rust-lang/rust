//! This test is a part of quality control and makes clippy eat what it produces. Awesome lints and
//! long error messages
//!
//! See [Eating your own dog food](https://en.wikipedia.org/wiki/Eating_your_own_dog_food) for context

#![feature(once_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use itertools::Itertools;
use std::path::PathBuf;
use std::process::Command;
use test_utils::IS_RUSTC_TEST_SUITE;

mod test_utils;

#[test]
fn dogfood_clippy() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let mut failed_packages = Vec::new();

    // "" is the root package
    for package in [
        "",
        "clippy_dev",
        "clippy_lints",
        "clippy_utils",
        "lintcheck",
        "rustc_tools_util",
    ] {
        if !run_clippy_for_package(package, &["-D", "clippy::all", "-D", "clippy::pedantic"]) {
            failed_packages.push(if package.is_empty() { "root" } else { package });
        }
    }

    assert!(
        !failed_packages.is_empty(),
        "Dogfood failed for packages `{}`",
        failed_packages.iter().format(", "),
    )
}

#[test]
#[ignore]
#[cfg(feature = "internal")]
fn run_metadata_collection_lint() {
    use std::fs::File;
    use std::time::SystemTime;

    // Setup for validation
    let metadata_output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("util/gh-pages/lints.json");
    let start_time = SystemTime::now();

    // Run collection as is
    std::env::set_var("ENABLE_METADATA_COLLECTION", "1");
    run_clippy_for_package("clippy_lints", &["-A", "unfulfilled_lint_expectations"]);

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
    run_clippy_for_package("clippy_lints", &["-A", "unfulfilled_lint_expectations"]);
}

fn run_clippy_for_package(project: &str, args: &[&str]) -> bool {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut command = Command::new(&*test_utils::CARGO_CLIPPY_PATH);

    command
        .current_dir(root_dir.join(project))
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features");

    if let Ok(dogfood_args) = std::env::var("__CLIPPY_DOGFOOD_ARGS") {
        for arg in dogfood_args.split_whitespace() {
            command.arg(arg);
        }
    }

    command.arg("--").args(args);
    command.arg("-Cdebuginfo=0"); // disable debuginfo to generate less data in the target dir

    if cfg!(feature = "internal") {
        // internal lints only exist if we build with the internal feature
        command.args(["-D", "clippy::internal"]);
    } else {
        // running a clippy built without internal lints on the clippy source
        // that contains e.g. `allow(clippy::invalid_paths)`
        command.args(["-A", "unknown_lints"]);
    }

    let output = command.output().unwrap();

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    output.status.success()
}
