// Functions with more than 6 arguments using foreign function interfaces (FFI) with C libraries
// would have their arguments unexpectedly swapped, causing unexpected behaviour in Rust-C FFI
// programs. This test compiles and executes Rust code with bulky functions of up to 7 arguments
// and uses assertions to check for unexpected swaps.
// See https://github.com/rust-lang/rust/issues/25594

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("test");
    rustc().input("test.rs").run();
    run("test");
}
