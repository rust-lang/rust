// `compiler-rt` ("runtime") is a suite of LLVM features compatible with rustc.
// After building it was enabled on Windows-gnu in #29874, this test is a basic smoke test to
// check if building and linking to it can work at all.
// See https://github.com/rust-lang/rust/pull/29478

//@ only-windows-gnu

use run_make_support::{cxx, llvm_ar, run, rustc, static_lib_name};

fn main() {
    cxx().input("foo.cpp").arg("-c").out_exe("foo.o").run();
    llvm_ar().obj_to_ar().output_input(static_lib_name("foo"), "foo.o").run();
    rustc().input("foo.rs").arg("-lfoo").arg("-lstdc++").run();
    run("foo");
}
