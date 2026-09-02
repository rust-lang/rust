//! Checks `--explain` against the forms of a lint name a user may actually type.
//!
//! `src/main.rs` lowercases the argument, strips a `clippy::` prefix and replaces
//! `-` with `_` before `clippy_lints::explain` looks the name up in its uppercase
//! form. Calling `explain` directly would leave that conversion untested.
//!
//! This test is a no-op if run as part of the compiler test suite
//! and will always succeed.

use std::process::{Command, Stdio};
use test_utils::{CARGO_CLIPPY_PATH, IS_RUSTC_TEST_SUITE};

mod test_utils;

#[test]
fn explain() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let running = [
        ("allow-attributes", true),
        ("clippy::allow_attributes", true),
        ("TOPLEVEL_REF_ARG", true),
        ("CLIPPY::TOPLEVEL_REF_ARG", true),
        ("not_a_lint", false),
    ]
    .map(|(arg, found)| {
        let child = Command::new(&*CARGO_CLIPPY_PATH)
            .args(["--explain", arg])
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        (arg, found, child)
    });

    for (arg, found, mut child) in running {
        assert_eq!(
            child.wait().unwrap().success(),
            found,
            "unexpected result for `--explain {arg}`"
        );
    }
}
