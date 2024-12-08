// Verify that no text relocations are accidentally introduced by linking a
// minimal rust staticlib.
// The test links a rust static library into a shared library, and checks that
// the linker doesn't have to flag the resulting file as containing TEXTRELs.
// This bug otherwise breaks Android builds, which forbid TEXTRELs.
// See https://github.com/rust-lang/rust/issues/68794

//@ ignore-cross-compile
//@ ignore-windows
// Reason: There is no `bar.dll` produced by CC to run readobj on

use run_make_support::{
    cc, dynamic_lib_name, extra_c_flags, extra_cxx_flags, llvm_readobj, rustc, static_lib_name,
};

fn main() {
    rustc().input("foo.rs").run();
    cc().input("bar.c")
        .input(static_lib_name("foo"))
        .out_exe(&dynamic_lib_name("bar"))
        .arg("-fPIC")
        .arg("-shared")
        .args(extra_c_flags())
        .args(extra_cxx_flags())
        .run();
    llvm_readobj()
        .input(dynamic_lib_name("bar"))
        .arg("--dynamic")
        .run()
        .assert_stdout_not_contains("TEXTREL");
}
