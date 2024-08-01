// During a forced unwind, crossing the non-Plain Old Frame
// would define the forced unwind as undefined behaviour, and
// immediately abort the unwinding process. This test checks
// that the forced unwinding takes precedence.
// See https://github.com/rust-lang/rust/issues/101469

//@ ignore-cross-compile
//@ ignore-windows
//Reason: pthread (POSIX threads) is not available on Windows

use run_make_support::{run, rustc};

fn main() {
    rustc().input("foo.rs").run();
    run("foo").assert_stdout_not_contains("cannot unwind");
}
