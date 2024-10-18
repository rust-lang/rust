use std::path::Path;

use run_make_support::rfs::remove_file;
use run_make_support::{Rustc, rustc};

fn run_rustc() -> Rustc {
    let mut rustc = rustc();
    rustc.arg("main.rs").output("main").linker("./fake-linker");
    rustc
}

fn main() {
    // first, compile our linker
    rustc().arg("fake-linker.rs").output("fake-linker").run();

    // Run rustc with our fake linker, and make sure it shows warnings
    let warnings = run_rustc().link_arg("run_make_warn").run();
    warnings.assert_stderr_contains("warning: linker stderr: bar");

    // Make sure it shows stdout, but only when --verbose is passed
    run_rustc()
        .link_arg("run_make_info")
        .verbose()
        .run()
        .assert_stderr_contains("warning: linker stdout: foo");
    run_rustc()
        .link_arg("run_make_info")
        .run()
        .assert_stderr_not_contains("warning: linker stdout: foo");

    // Make sure we short-circuit this new path if the linker exits with an error
    // (so the diagnostic is less verbose)
    run_rustc().link_arg("run_make_error").run_fail().assert_stderr_contains("note: error: baz");

    // Make sure we don't show the linker args unless `--verbose` is passed
    run_rustc()
        .link_arg("run_make_error")
        .verbose()
        .run_fail()
        .assert_stderr_contains_regex("fake-linker.*run_make_error");
    run_rustc()
        .link_arg("run_make_error")
        .run_fail()
        .assert_stderr_not_contains_regex("fake-linker.*run_make_error");
}
