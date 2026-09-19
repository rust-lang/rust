// When calling `rustc` without an explicit edition, emit a note asking the user to specify one,
// clarifying that the default is 2015.

use run_make_support::{bare_rustc, diff, rustc};

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
    bare_rustc().arg("--version").run().assert_stderr_not_contains("--edition");
}
