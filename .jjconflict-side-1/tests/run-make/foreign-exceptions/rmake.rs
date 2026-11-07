// This test was created to check that compilation and execution still works
// after the addition of a new feature, in #65646: the ability to unwind panics
// and exceptions back and forth between Rust and C++. This is a basic smoke test,
// this feature being broken in quiet or subtle ways could still result in this test
// passing.
// See https://github.com/rust-lang/rust/pull/65646

//@ needs-unwind
// Reason: this test exercises panic unwinding
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib_cxx, run, rustc};

fn main() {
    build_native_static_lib_cxx("foo");
    rustc().input("foo.rs").arg("-lfoo").extra_rs_cxx_flags().run();
    run("foo");
}
