use crate::utils::exit_if_err;
use std::process::Command;

/// # Panics
///
/// Panics if unable to run the dogfood test
#[allow(clippy::fn_params_excessive_bools)]
pub fn dogfood(fix: bool, allow_dirty: bool, allow_staged: bool, allow_no_vcs: bool) {
    let mut cmd = Command::new("cargo");

    cmd.args(["test", "--test", "dogfood"])
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

    if allow_no_vcs {
        dogfood_args.push("--allow-no-vcs");
    }

    cmd.env("__CLIPPY_DOGFOOD_ARGS", dogfood_args.join(" "));

    exit_if_err(cmd.status());
}
