// In this test, the functions foo() and bar() must avoid being mangled, as
// the external C function depends on them to return the correct sum of 3 + 5 = 8.
// This test therefore checks that the compiled and executed program respects the
// #[no_mangle] flags successfully.
// See https://github.com/rust-lang/rust/pull/15831

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("test.rs").run();
    run("test");
}
