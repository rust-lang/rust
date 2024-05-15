// This test generates an instrumented binary - a program which
// will keep track of how many times it calls each function, a useful
// feature for optimization. Then, an argument (aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc)
// is passed into the instrumented binary, which should react with a number of function calls
// fully known in advance. (For example, the letter 'a' results in calling f1())

// If the test passes, the expected function call count was added to the use-phase LLVM-IR.
// See https://github.com/rust-lang/rust/pull/66631

//@ needs-profiler-support
//@ ignore-cross-compile

// (This test has problems generating profdata on mingw. This could use further investigation.)
//@ ignore-windows-gnu

use run_make_support::{
    llvm_filecheck, llvm_profdata, run_with_args, rustc, rustdoc, target, tmp_dir,
};
use std::fs;

fn main() {
    let path_prof_data_dir = tmp_dir().join("prof_data_dir");
    let path_merged_profdata = path_prof_data_dir.join("merged.profdata");
    rustc().input("opaque.rs").run();
    fs::create_dir_all(&path_prof_data_dir);
    rustc()
        .input("interesting.rs")
        .profile_generate(&path_prof_data_dir)
        .opt()
        .codegen_units(1)
        .run();
    rustc().input("main.rs").profile_generate(&path_prof_data_dir).opt().run();
    run_with_args("main", &["aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc"]);
    llvm_profdata()
        .merge()
        .output(&path_merged_profdata)
        .input(path_prof_data_dir)
        .command_output();
    rustc()
        .input("interesting.rs")
        .profile_use(path_merged_profdata)
        .opt()
        .codegen_units(1)
        .emit("llvm-ir")
        .run();
    let interesting_ll = tmp_dir().join("interesting.ll");
    llvm_filecheck().patterns("filecheck-patterns.txt").stdin(interesting_ll);
}
