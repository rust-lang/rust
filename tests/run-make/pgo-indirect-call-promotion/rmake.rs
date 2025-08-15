// This test checks that indirect call promotion is performed. The test
// programs calls the same function a thousand times through a function pointer.
// Only PGO data provides the information that it actually always is the same
// function. We verify that the indirect call promotion pass inserts a check
// whether it can make a direct call instead of the indirect call.
// See https://github.com/rust-lang/rust/pull/66631

//@ needs-profiler-runtime
// Reason: llvm_profdata is used
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{llvm_filecheck, llvm_profdata, rfs, run, rustc};

fn main() {
    // We don't compile `opaque` with either optimizations or instrumentation.
    rustc().input("opaque.rs").codegen_source_order().run();
    // Compile the test program with instrumentation
    rfs::create_dir("prof_data_dir");
    rustc()
        .input("interesting.rs")
        .profile_generate("prof_data_dir")
        .opt()
        .codegen_units(1)
        .codegen_source_order()
        .run();
    rustc().input("main.rs").profile_generate("prof_data_dir").opt().codegen_source_order().run();
    // The argument below generates to the expected branch weights
    run("main");
    llvm_profdata().merge().output("prof_data_dir/merged.profdata").input("prof_data_dir").run();
    rustc()
        .input("interesting.rs")
        .profile_use("prof_data_dir/merged.profdata")
        .opt()
        .codegen_units(1)
        .emit("llvm-ir")
        .codegen_source_order()
        .run();
    llvm_filecheck()
        .patterns("filecheck-patterns.txt")
        .stdin_buf(rfs::read("interesting.ll"))
        .run();
}
