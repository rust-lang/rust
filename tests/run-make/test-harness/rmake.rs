// The way test suites run can be modified using configuration flags,
// ignoring certain tests while running others. This test contains two
// functions, one which must run and the other which must not. The standard
// output is checked to verify that the ignore configuration is doing its job,
// and that output is successfully minimized with the --quiet flag.
// See https://github.com/rust-lang/rust/commit/f7ebe23ae185991b0fee05b32fbb3e29b89a41bf

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{run, run_with_args, rustc};

fn main() {
    rustc().arg("--test").input("test-ignore-cfg.rs").cfg("ignorecfg").run();
    // check that #[cfg_attr(..., ignore)] does the right thing.
    run("test-ignore-cfg")
        .assert_stdout_contains("shouldnotignore ... ok")
        .assert_stdout_contains("shouldignore ... ignored");
    assert_eq!(
        // One of the lines is exactly "i."
        run_with_args("test-ignore-cfg", &["--quiet"]).stdout_utf8().lines().find(|&x| x == "i."),
        Some("i.")
    );
    run_with_args("test-ignore-cfg", &["--quiet"]).assert_stdout_not_contains("should");
}
