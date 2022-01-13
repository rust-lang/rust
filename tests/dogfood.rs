//! This test is a part of quality control and makes clippy eat what it produces. Awesome lints and
//! long error messages
//!
//! See [Eating your own dog food](https://en.wikipedia.org/wiki/Eating_your_own_dog_food) for context

#![feature(once_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::path::PathBuf;
use std::process::Command;
use test_utils::IS_RUSTC_TEST_SUITE;

mod test_utils;

#[test]
fn dogfood_clippy() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    // "" is the root package
    for package in &["", "clippy_dev", "clippy_lints", "clippy_utils", "rustc_tools_util"] {
        run_clippy_for_package(package);
    }
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
    run_clippy_for_package("clippy_lints");

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
    run_clippy_for_package("clippy_lints");
}

fn run_clippy_for_package(project: &str) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut command = Command::new(&*test_utils::CARGO_CLIPPY_PATH);

    command
        .current_dir(root_dir.join(project))
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::pedantic"])
        .arg("-Cdebuginfo=0"); // disable debuginfo to generate less data in the target dir

    // internal lints only exist if we build with the internal feature
    if cfg!(feature = "internal") {
        command.args(&["-D", "clippy::internal"]);
    }

    let output = command.output().unwrap();

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}
