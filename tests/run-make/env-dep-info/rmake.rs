//@ needs-target-std
//
// Inside dep-info emit files, #71858 made it so all accessed environment
// variables are usefully printed. This test checks that this feature works
// as intended by checking if the environment variables used in compilation
// appear in the output dep-info files.
// See https://github.com/rust-lang/rust/issues/40364

use run_make_support::{diff, rustc};

fn main() {
    rustc()
        .env("EXISTING_ENV", "1")
        .env("EXISTING_OPT_ENV", "1")
        .emit("dep-info")
        .input("main.rs")
        .run();
    diff().expected_file("correct_main.d").actual_file("main.d").run();
    // Procedural macro
    rustc().input("macro_def.rs").run();
    rustc().env("EXISTING_PROC_MACRO_ENV", "1").emit("dep-info").input("macro_use.rs").run();
    diff().expected_file("correct_macro.d").actual_file("macro_use.d").run();
}
