// Test that C++ catch-all blocks can successfully catch Rust-originated panics.

//@ ignore-cross-compile
// Reason: the compiled binary is executed

//@ ignore-wasm32
//@ ignore-wasm64
// Reason: compiling C++ to WASM may cause problems.

// Not exercised in full CI, but sgx technically supports std.
//@ ignore-sgx

use run_make_support::{build_native_static_lib_cxx, run, rustc};

fn main() {
    build_native_static_lib_cxx("catch");
    rustc().input("panic.rs").arg("-lcatch").extra_rs_cxx_flags().run();
    run("panic");
}
