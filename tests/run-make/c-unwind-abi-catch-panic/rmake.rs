// A test for calling `C-unwind` functions across foreign function boundaries (FFI).
// This test triggers a panic when calling a foreign function that calls *back* into Rust.
// This catches a panic across an FFI boundary and downcasts it into an integer.
// The Rust code that panics is in the same directory, unlike `c-unwind-abi-catch-lib-panic`.
// See https://github.com/rust-lang/rust/pull/76570

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ needs-unwind
// Reason: this test exercises panic unwinding

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("add");
    rustc().input("main.rs").run();
    run("main");
}
