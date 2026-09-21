// When calling `rustc` without an explicit edition, emit a note asking the user to specify one,
// clarifying that the default is 2015.

use run_make_support::{bare_rustc, diff, rustc, rustdoc};

fn main() {
    rustc().edition("2015").input("main.rs").run().assert_stderr_not_contains("--edition");
    let out = rustc().input("main.rs").run().assert_stderr_contains("--edition").stderr_utf8();
    diff().expected_file("unspecified-edition.stderr").actual_text("(rustc)", &out).run();

    // Ensure that we only mention --edition when compiling code.
    let out = rustc().run_fail().assert_stderr_not_contains("--edition").stderr_utf8();
    diff()
        .expected_file("unspecified-edition-without-compiling.stderr")
        .actual_text("(rustc)", &out)
        .run();

    // Ensure that we dont mention --edition when running rustdoc.
    let out = rustdoc().run_fail().assert_stderr_not_contains("--edition").stderr_utf8();
    diff()
        .expected_text("(test)", "error: missing file operand\n\n")
        .actual_text("(rustc)", &out)
        .run();

    let out =
        rustdoc().input("main.rs").run().assert_stderr_not_contains("--edition").stderr_utf8();
    diff().expected_text("(test)", "").actual_text("(rustc)", &out).run();

    // Ensure that we don't mention --edition when getting help.
    let result = rustc().arg("--help").run();
    result.assert_stderr_not_contains("--edition");
    let out = result.stdout_utf8();
    let err = result.stderr_utf8();
    diff().expected_file("help-unspecified-edition.stdout").actual_text("(rustc)", &out).run();
    diff().expected_text("(test)", "").actual_text("(rustc)", &err).run();
    bare_rustc().arg("--version").run().assert_stderr_not_contains("--edition");
}
