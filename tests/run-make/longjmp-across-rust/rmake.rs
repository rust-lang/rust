// longjmp, an error handling function used in C, is useful
// for jumping out of nested call chains... but it used to accidentally
// trigger Rust's cleanup system in a way that caused an unexpected abortion
// of the program. After this was fixed in #48572, this test compiles and executes
// a program that jumps between Rust and its C library, with longjmp included. For
// the test to succeed, no unexpected abortion should occur.
// See https://github.com/rust-lang/rust/pull/48572

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("foo");
    rustc().input("main.rs").run();
    run("main");
}
