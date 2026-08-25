// Packed structs, in C, occupy less bytes in memory, but are more
// vulnerable to alignment errors. Passing them around in a Rust-C foreign
// function interface (FFI) would cause unexpected behavior, until this was
// fixed in #16584. This test checks that a Rust program with a C library
// compiles and executes successfully, even with usage of a packed struct.
// See https://github.com/rust-lang/rust/issues/16574

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("test.rs").run();
    run("test");
}
