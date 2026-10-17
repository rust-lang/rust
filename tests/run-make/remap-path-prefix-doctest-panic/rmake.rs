// Checks that the location printed by a panicking doctest is remapped
// according to `--remap-path-scope`.
//
// Only the lines between the start of the stack backtrace and the end are
// are inspected, so timings, temp dirs and backtraces can't leak into the check.

//@ ignore-cross-compile
//@ ignore-windows

use run_make_support::assertion_helpers::{assert_contains, assert_not_contains};
use run_make_support::rustdoc;

/// Runs the failing doctest and returns stdout.
fn panic_file(scope: Option<&str>) -> String {
    let mut cmd = rustdoc();
    cmd.input("src/doctest.rs")
        .arg("--test")
        .arg("--remap-path-prefix=src=/REMAPPED")
        .arg("-Cdebuginfo=1")
        .arg("--edition=2015")
        .env("RUST_BACKTRACE", "1");

    if let Some(scope) = scope {
        cmd.arg(format!("--remap-path-scope={scope}")).arg("-Zunstable-options");
    }

    let out = cmd.run_fail();
    out.assert_stdout_contains("explicit panic");

    let stdout = out.stdout_utf8();
    let section: Vec<&str> = stdout
        .lines()
        .skip_while(|l| l.trim() != "stack backtrace:")
        .skip(1)
        .take_while(|l| l.trim() != "failures:")
        .collect();

    assert!(!section.is_empty(), "no backtrace found in output:\n{stdout}");
    section.join("\n")
}

fn main() {
    // Check that with debuginfo (and related) we get the remapped path
    for scope in [Some("debuginfo"), Some("object"), Some("all"), None] {
        let stdout = panic_file(scope);
        assert_contains(&stdout, "/REMAPPED/doctest.rs");
        assert_not_contains(&stdout, "src/doctest.rs");
    }

    for scope in ["diagnostics", "documentation", "macro"] {
        let stdout = panic_file(Some(scope));
        assert_contains(&stdout, "src/doctest.rs");
        assert_not_contains(&stdout, "/REMAPPED/doctest.rs");
    }
}
