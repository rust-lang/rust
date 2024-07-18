// Some start files were missed when originally writing the logic to swap in musl start files.
// This caused #36710. After the fix in #50105, this test checks that linking to C++ code
// with global destructors works.
// See https://github.com/rust-lang/rust/pull/50105

//@ ignore-cross-compile
// Reason: the compiled binary is executed

// FIXME(Oneirical): are these really necessary? This test is supposed to test a musl
// bug... and it ignores musl? This wasn't part of the original test at its creation, which
// had no ignores.

//# ignore-none no-std is not supported
//# ignore-wasm32 FIXME: don't attempt to compile C++ to WASM
//# ignore-wasm64 FIXME: don't attempt to compile C++ to WASM
//# ignore-nvptx64-nvidia-cuda FIXME: can't find crate for `std`
//# ignore-musl FIXME: this makefile needs teaching how to use a musl toolchain
//#                    (see dist-i586-gnu-i586-i686-musl Dockerfile)
//# ignore-sgx

use run_make_support::{build_native_static_lib_cxx, run, rustc};

fn main() {
    build_native_static_lib_cxx("foo");
    rustc().input("foo.rs").arg("-lfoo").extra_rs_cxx_flags().run();
    run("foo");
}
