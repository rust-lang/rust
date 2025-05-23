use crate::utils::{cargo_cmd, run_exit_on_err};
use itertools::Itertools;

/// # Panics
///
/// Panics if unable to run the dogfood test
#[allow(clippy::fn_params_excessive_bools)]
pub fn dogfood(fix: bool, allow_dirty: bool, allow_staged: bool, allow_no_vcs: bool) {
    run_exit_on_err(
        "cargo test",
        cargo_cmd()
            .args(["test", "--test", "dogfood"])
            .args(["--features", "internal"])
            .args(["--", "dogfood_clippy", "--nocapture"])
            .env(
                "__CLIPPY_DOGFOOD_ARGS",
                [
                    if fix { "--fix" } else { "" },
                    if allow_dirty { "--allow-dirty" } else { "" },
                    if allow_staged { "--allow-staged" } else { "" },
                    if allow_no_vcs { "--allow-no-vcs" } else { "" },
                ]
                .iter()
                .filter(|x| !x.is_empty())
                .join(" "),
            ),
    );
}
