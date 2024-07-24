// This test was created in response to an obscure miscompilation bug, only
// visible with the -O3 flag passed to the cc compiler when trying to obtain
// a native static library for the sake of foreign function interface. This
// flag could cause certain integer types to fail to be zero-extended, resulting
// in type casting errors. After the fix in #97800, this test attempts integer casting
// while simultaneously interfacing with a C library and using the -O3 flag.
// See https://github.com/rust-lang/rust/issues/97463

//@ ignore-msvc
// Reason: the rustc compilation fails due to an unresolved external symbol

//@ ignore-cross-compile
// Reason: The compiled binary is executed.

use run_make_support::{cc, is_msvc, llvm_ar, run, rustc, static_lib_name};

fn main() {
    // The issue exercised by this test specifically needs needs `-O`
    // flags (like `-O3`) to reproduce. Thus, we call `cc()` instead of
    // the nicer `build_native_static_lib`.
    cc().arg("-c").arg("-O3").out_exe("bad.o").input("bad.c").run();
    llvm_ar().obj_to_ar().output_input(static_lib_name("bad"), "bad.o").run();
    rustc().input("param_passing.rs").arg("-lbad").opt_level("3").run();
    run("param_passing");
}
