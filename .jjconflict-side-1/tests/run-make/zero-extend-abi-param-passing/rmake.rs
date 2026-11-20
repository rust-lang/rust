// This test was created in response to an obscure miscompilation bug, only
// visible with the -O3 flag passed to the cc compiler when trying to obtain
// a native static library for the sake of foreign function interface. This
// flag could cause certain integer types to fail to be zero-extended, resulting
// in type casting errors. After the fix in #97800, this test attempts integer casting
// while simultaneously interfacing with a C library and using the -O3 flag.
// See https://github.com/rust-lang/rust/issues/97463

//@ ignore-cross-compile
// Reason: The compiled binary is executed.
use run_make_support::{build_native_static_lib_optimized, run, rustc};

fn main() {
    // The issue exercised by this test specifically needs an optimized native static lib.
    build_native_static_lib_optimized("bad");
    rustc().input("param_passing.rs").opt_level("3").run();
    run("param_passing");
}
