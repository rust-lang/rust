// This test generates an instrumented binary - a program which
// will keep track of how many times it calls each function, a useful
// feature for optimization. Then, an argument (aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc)
// is passed into the instrumented binary, which should react with a number of function calls
// fully known in advance. (For example, the letter 'a' results in calling f1())

// If the test passes, the expected function call count was added to the use-phase LLVM-IR.
// See https://github.com/rust-lang/rust/pull/66631

//@ needs-profiler-runtime
//@ ignore-cross-compile

use std::path::Path;

use run_make_support::{llvm_filecheck, llvm_profdata, rfs, run_with_args, rustc};

fn main() {
    let path_prof_data_dir = Path::new("prof_data_dir");
    let path_merged_profdata = path_prof_data_dir.join("merged.profdata");
    rustc().input("opaque.rs").codegen_source_order().run();
    rfs::create_dir_all(&path_prof_data_dir);
    rustc()
        .input("interesting.rs")
        .profile_generate(&path_prof_data_dir)
        .opt()
        .codegen_units(1)
        .codegen_source_order()
        .run();
    rustc()
        .input("main.rs")
        .profile_generate(&path_prof_data_dir)
        .opt()
        .codegen_source_order()
        .run();
    run_with_args("main", &["aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc"]);
    llvm_profdata().merge().output(&path_merged_profdata).input(path_prof_data_dir).run();
    rustc()
        .input("interesting.rs")
        .profile_use(path_merged_profdata)
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
