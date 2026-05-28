// Slices were broken when implicated in foreign-function interface (FFI) with
// a C library, with something as simple as measuring the length or returning
// an item at a certain index of a slice would cause an internal compiler error (ICE).
// This was fixed in #25653, and this test checks that slices in Rust-C FFI can be part
// of a program that compiles and executes successfully.
// See https://github.com/rust-lang/rust/issues/25581

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("test.rs").run();
    run("test");
}
