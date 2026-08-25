// This test makes sure that PGO profiling data leads to cold functions being
// marked as `cold` and hot functions with `inlinehint`.
// The test program contains an `if` where actual execution only ever takes the
// `else` branch. Accordingly, we expect the function that is never called to
// be marked as cold.
// See https://github.com/rust-lang/rust/pull/60262

//@ needs-profiler-runtime
//@ ignore-cross-compile

use run_make_support::{
    cwd, has_extension, has_prefix, llvm_filecheck, llvm_profdata, rfs, run_with_args, rustc,
    shallow_find_files,
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
        .codegen_source_order()
        .profile_generate(cwd())
        .input("main.rs")
        .run();
    // Run it in order to generate some profiling data
    run_with_args("main", &["some-argument"]);
    // Postprocess the profiling data so it can be used by the compiler
    let profraw_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "default") && has_extension(path, "profraw")
    });
    let profraw_file = profraw_files.get(0).unwrap();
    llvm_profdata().merge().output("merged.profdata").input(profraw_file).run();
    // Compile the test program again, making use of the profiling data
    rustc()
        .opt_level("2")
        .codegen_units(1)
        .arg("-Cllvm-args=-disable-preinline")
        .profile_use("merged.profdata")
        .emit("llvm-ir")
        .codegen_source_order()
        .input("main.rs")
        .run();
    // Check that the generate IR contains some things that we expect.
    // We feed the file into LLVM FileCheck tool *with its lines reversed* so that we see the
    // line with the function name before the line with the function attributes.
    // FileCheck only supports checking that something matches on the next line,
    // but not if something matches on the previous line.
    let ir = rfs::read_to_string("main.ll");
    let lines: Vec<_> = ir.lines().rev().collect();
    let mut reversed_ir = lines.join("\n");
    reversed_ir.push('\n');
    llvm_filecheck().patterns("filecheck-patterns.txt").stdin_buf(reversed_ir.as_bytes()).run();
}
