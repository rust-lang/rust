// Generic types in foreign-function interfaces were introduced in #15831 - this
// test simply runs a Rust program containing generics that is also reliant on
// a C library, and checks that compilation and execution are successful.
// See https://github.com/rust-lang/rust/pull/15831

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("testcrate.rs").run();
    rustc().input("test.rs").run();
    run("test");
}
