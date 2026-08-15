// Despite the absence of any unsafe Rust code, foo.rs in this test would,
// because of the raw function pointer,
// cause undefined behavior and fail to print the expected result, "4" -
// only when activating optimizations (opt-level 2). This test checks
// that this bug does not make a resurgence.
// Note that the bug cannot be observed in an assert_eq!, only in the stdout.
// See https://github.com/rust-lang/rust/issues/20626

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().input("foo.rs").opt().run();
    run("foo").assert_stdout_equals("4");
}
