use crate::{clippy_project_root, exit_if_err};
use std::process::Command;

/// # Panics
///
/// Panics if unable to run the dogfood test
pub fn dogfood(fix: bool, allow_dirty: bool, allow_staged: bool) {
    let mut cmd = Command::new("cargo");

    cmd.current_dir(clippy_project_root())
        .args(["test", "--test", "dogfood"])
        .args(["--features", "internal"])
        .args(["--", "dogfood_clippy", "--nocapture"]);

    let mut dogfood_args = Vec::new();
    if fix {
        dogfood_args.push("--fix");
    }

    if allow_dirty {
        dogfood_args.push("--allow-dirty");
    }

    if allow_staged {
        dogfood_args.push("--allow-staged");
    }

    cmd.env("__CLIPPY_DOGFOOD_ARGS", dogfood_args.join(" "));

    exit_if_err(cmd.status());
}
