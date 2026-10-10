// During unwinding, an implementation of Drop is possible to clean up resources.
// This test implements drop in both a main function and its static library.
// If the test succeeds, a Rust program being a static library does not affect Drop implementations.
// See https://github.com/rust-lang/rust/issues/10434

//@ ignore-cross-compile
//@ needs-unwind

use run_make_support::{run, rustc};

fn main() {
    rustc().input("lib.rs").run();
    rustc().input("main.rs").run();
    run("main");
}
