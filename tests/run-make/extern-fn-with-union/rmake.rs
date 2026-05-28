// If an external function from foreign-function interface was called upon,
// its attributes would only be passed to LLVM if and only if it was called in the same crate.
// This caused passing around unions to be incorrect.
// See https://github.com/rust-lang/rust/pull/14191

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("ctest");
    rustc().input("testcrate.rs").run();
    rustc().input("test.rs").run();
    run("test");
}
