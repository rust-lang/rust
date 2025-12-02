//@ ignore-cross-compile (needs to run doctests)

use run_make_support::rfs::write;
use run_make_support::{cwd, rustdoc};

fn assert_presence_of_compilation_time_report(
    content: &str,
    success: bool,
    should_contain_compile_time: bool,
) {
    let mut cmd = rustdoc();
    let file = cwd().join("foo.rs");

    write(&file, content);
    cmd.input(&file).arg("--test").edition("2024").env("RUST_BACKTRACE", "0");
    let output = if success { cmd.run() } else { cmd.run_fail() };

    assert_eq!(
        output
            .stdout_utf8()
            .split("all doctests ran in ")
            .last()
            .is_some_and(|s| s.contains("; merged doctests compilation took")),
        should_contain_compile_time,
    );
}

fn main() {
    // Checking with only successful merged doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! let x = 12;
//! ```",
        true,
        true,
    );
    // Checking with only failing merged doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! panic!();
//! ```",
        false,
        true,
    );
    // Checking with mix of successful doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! let x = 12;
//! ```
//!
//! ```compile_fail
//! let x
//! ```",
        true,
        true,
    );
    // Checking with mix of failing doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! panic!();
//! ```
//!
//! ```compile_fail
//! let x
//! ```",
        false,
        true,
    );
    // Checking with mix of failing doctests (v2).
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! let x = 12;
//! ```
//!
//! ```compile_fail
//! let x = 12;
//! ```",
        false,
        true,
    );
    // Checking with mix of failing doctests (v3).
    assert_presence_of_compilation_time_report(
        "\
//! ```
//! panic!();
//! ```
//!
//! ```compile_fail
//! let x = 12;
//! ```",
        false,
        true,
    );
    // Checking with successful non-merged doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```compile_fail
//! let x
//! ```",
        true,
        // If there is no merged doctests, then we should not display compilation time.
        false,
    );
    // Checking with failing non-merged doctests.
    assert_presence_of_compilation_time_report(
        "\
//! ```compile_fail
//! let x = 12;
//! ```",
        false,
        // If there is no merged doctests, then we should not display compilation time.
        false,
    );
}
