// This test makes sure that PGO profiling data leads to cold functions being
// marked as `cold` and hot functions with `inlinehint`.
// The test program contains an `if` where actual execution only ever takes the
// `else` branch. Accordingly, we expect the function that is never called to
// be marked as cold.
// See https://github.com/rust-lang/rust/pull/60262

//@ needs-profiler-support
//@ ignore-cross-compile

use run_make_support::{
    cwd, find_files_by_prefix_and_extension, fs_wrapper, llvm_filecheck, llvm_profdata,
    run_with_args, rustc,
};

fn main() {
    // Compile the test program with instrumentation
    // Disable the pre-inlining pass (i.e. a pass that does some inlining before
    // it adds the profiling instrumentation). Disabling this pass leads to
    // rather predictable IR which we need for this test to be stable.
    rustc()
        .opt_level("2")
        .codegen_units(1)
        .arg("-Cllvm-args=-disable-preinline")
        .profile_generate(cwd())
        .input("main.rs")
        .run();
    // Run it in order to generate some profiling data
    run_with_args("main", &["some-argument"]);
    // Postprocess the profiling data so it can be used by the compiler
    llvm_profdata()
        .merge()
        .output("merged.profdata")
        .input(find_files_by_prefix_and_extension(cwd(), "default", "profraw").get(0).unwrap())
        .run();
    // Compile the test program again, making use of the profiling data
    rustc()
        .opt_level("2")
        .codegen_units(1)
        .arg("-Cllvm-args=-disable-preinline")
        .profile_use("merged.profdata")
        .emit("llvm-ir")
        .input("main.rs")
        .run();
    // Check that the generate IR contains some things that we expect
    //
    // We feed the file into LLVM FileCheck tool *in reverse* so that we see the
    // line with the function name before the line with the function attributes.
    // FileCheck only supports checking that something matches on the next line,
    // but not if something matches on the previous line.
    let mut bytes = fs_wrapper::read("interesting.ll");
    bytes.reverse();
    llvm_filecheck().patterns("filecheck-patterns.txt").stdin(bytes).run();
}
