// Regression test for https://github.com/rust-lang/rust/issues/88426.
use run_make_support::{bare_rustdoc, diff};

fn main() {
    for bootstrap in ["-1", "1"] {
        let output = bare_rustdoc().env("RUSTC_BOOTSTRAP", bootstrap).arg("--help").run();

        output.assert_stderr_equals("");
        let help = output.stdout_utf8();

        let no_run = help.lines().find(|line| line.trim_start().starts_with("--no-run ")).unwrap();
        assert!(no_run.contains("(unstable)"));

        let help_option = help.lines().find(|line| line.contains("--help ")).unwrap();
        assert!(!help_option.contains("(unstable)"));
    }

    // Simulate stable restrictions: requesting an unusable opt-in is misleading.
    let output =
        bare_rustdoc().env("RUSTC_BOOTSTRAP", "-1").args(["--no-run", "--help"]).run_fail();

    output.assert_exit_code(1);
    output
        .assert_stderr_contains("the option `no-run` is only accepted on the nightly compiler")
        .assert_stderr_not_contains("must also be passed");

    diff()
        .expected_file("requires-nightly.stderr")
        .actual_text("rustdoc stderr", output.stderr_utf8())
        .run();

    // Allow unstable options, but do not supply the required opt-in.
    let output = bare_rustdoc().env("RUSTC_BOOTSTRAP", "1").args(["--no-run", "--help"]).run_fail();

    output.assert_exit_code(1);
    output.assert_stderr_contains("`-Z unstable-options` flag must also be passed");

    diff()
        .expected_file("missing-opt-in.stderr")
        .actual_text("rustdoc stderr", output.stderr_utf8())
        .run();

    // With permission and opt-in, argument checking should succeed.
    let output = bare_rustdoc()
        .env("RUSTC_BOOTSTRAP", "1")
        .args(["--no-run", "-Zunstable-options", "--help"])
        .run();

    output.assert_stderr_equals("");
}
