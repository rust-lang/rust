//@ ignore-cross-compile (needs to run host tool binary)

// Test the behavior of doctests that are marked `test_harness` and that contain multiple `#[test]`
// functions. Sadly this needs to be a run-make test instead of rustdoc-UI one because at the time
// of writing we can only pass `--test-threads=1` to the inner test suite using a runtool (needed to
// guarantee deterministic test output) which we'd like to be written in Rust to be cross-platform.
//
// See also #157511 and `tests/rustdoc-ui/doctest/test_harness.rs`.

use std::path::Path;

use run_make_support::{diff, rustc, rustdoc};

fn main() {
    let doctests_path = Path::new("doctests.rs");
    let runtool_path = Path::new("runtool.rs");

    rustc().input(doctests_path).crate_type("lib").run();
    rustc().input(runtool_path).run();

    let output = rustdoc()
        .edition("2015")
        .input(doctests_path)
        .arg("--test")
        // for the outer test suite
        .arg("--test-args=--test-threads=1")
        .arg("--test-runtool")
        .arg(Path::new(".").join(runtool_path).with_extension(std::env::consts::EXE_EXTENSION))
        .arg("-L.")
        .env("RUST_BACKTRACE", "0")
        .run_fail();
    output.assert_exit_code(101);
    output.assert_stderr_equals("");

    diff()
        .expected_file(doctests_path.with_extension("stdout"))
        .actual_text("stdout", output.stdout_utf8())
        .normalize(r#"finished in \d+\.\d+s"#, "finished in $$TIME")
        .normalize(r"thread '(?P<name>.*?)' \(\d+\) panicked", "thread '$name' ($$TID) panicked")
        .normalize(r"Test executable failed \(.+?\)", "Test executable failed ($$STATUS)")
        .run();
}
