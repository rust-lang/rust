//! This test is a part of quality control and makes clippy eat what it produces. Awesome lints and
//! long error messages
//!
//! See [Eating your own dog food](https://en.wikipedia.org/wiki/Eating_your_own_dog_food) for context

#![warn(rust_2018_idioms, unused_lifetimes)]

use itertools::Itertools;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::process::Command;
use test_utils::IS_RUSTC_TEST_SUITE;
use ui_test::Args;

mod test_utils;

fn main() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let args = Args::test().unwrap();

    if args.list {
        if !args.ignored {
            println!("dogfood: test");
        }
    } else if !args.skip.iter().any(|arg| arg == "dogfood") {
        dogfood();
    }
}

fn dogfood() {
    let mut failed_packages = Vec::new();

    for package in [
        "./",
        "clippy_dev",
        "clippy_lints_internal",
        "clippy_lints",
        "clippy_utils",
        "clippy_config",
        "lintcheck",
        "rustc_tools_util",
    ] {
        println!("linting {package}");
        if !run_clippy_for_package(package, &["-D", "clippy::all", "-D", "clippy::pedantic"]) {
            failed_packages.push(if package.is_empty() { "root" } else { package });
        }
    }

    assert!(
        failed_packages.is_empty(),
        "Dogfood failed for packages `{}`",
        failed_packages.iter().join(", "),
    );
}

#[must_use]
fn run_clippy_for_package(project: &str, args: &[&str]) -> bool {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut command = Command::new(&*test_utils::CARGO_CLIPPY_PATH);

    command
        .current_dir(root_dir.join(project))
        .env("CARGO_INCREMENTAL", "0")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features");

    if !io::stdout().is_terminal() {
        command.arg("-q");
    }

    if let Ok(dogfood_args) = std::env::var("__CLIPPY_DOGFOOD_ARGS") {
        for arg in dogfood_args.split_whitespace() {
            command.arg(arg);
        }
    }

    command.arg("--").args(args);
    command.arg("-Cdebuginfo=0"); // disable debuginfo to generate less data in the target dir
    command.args(["-D", "clippy::dbg_macro"]);

    if !cfg!(feature = "internal") {
        // running a clippy built without internal lints on the clippy source
        // that contains e.g. `allow(clippy::invalid_paths)`
        command.args(["-A", "unknown_lints"]);
    }

    command.status().unwrap().success()
}
