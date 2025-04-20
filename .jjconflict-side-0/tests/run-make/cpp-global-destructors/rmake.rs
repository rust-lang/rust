// Some start files were missed when originally writing the logic to swap in musl start files.
// This caused #36710. After the fix in #50105, this test checks that linking to C++ code
// with global destructors works.
// See https://github.com/rust-lang/rust/pull/50105

//@ ignore-cross-compile
// Reason: the compiled binary is executed

//@ ignore-wasm32
//@ ignore-wasm64
// Reason: compiling C++ to WASM may cause problems.

// Not exercised in full CI, but sgx technically supports std.
//@ ignore-sgx

use run_make_support::{build_native_static_lib_cxx, run, rustc};

fn main() {
    build_native_static_lib_cxx("foo");
    rustc().input("foo.rs").arg("-lfoo").extra_rs_cxx_flags().run();
    run("foo");
}
