// Static variables coming from a C library through foreign function interface (FFI) are unsized
// at compile time - and assuming they are sized used to cause an internal compiler error (ICE).
// After this was fixed in #58192, this test checks that external statics can be safely used in
// a program that both compiles and executes successfully.
// See https://github.com/rust-lang/rust/issues/57876

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("define-foo");
    rustc().arg("-ldefine-foo").input("use-foo.rs").run();
    run("use-foo");
}
