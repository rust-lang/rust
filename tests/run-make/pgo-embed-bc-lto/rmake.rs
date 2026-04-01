// This test case verifies that we successfully complete an LTO build with PGO
// using the embedded bitcode.
// It also ensures that the generated IR correctly includes the call results.

//@ needs-profiler-runtime
//@ ignore-cross-compile

use std::path::Path;

use run_make_support::{
    has_extension, has_prefix, llvm_filecheck, llvm_profdata, rfs, run, rustc, shallow_find_files,
};

fn run_test(cg_units: usize) {
    let path_prof_data_dir = Path::new("prof_data_dir");
    if path_prof_data_dir.exists() {
        rfs::remove_dir_all(path_prof_data_dir);
    }
    rfs::create_dir_all(&path_prof_data_dir);
    let path_merged_profdata = path_prof_data_dir.join("merged.profdata");
    rustc().input("opaque.rs").codegen_units(1).run();
    rustc()
        .input("interesting.rs")
        .profile_generate(&path_prof_data_dir)
        .opt()
        .crate_type("lib,cdylib")
        .codegen_units(cg_units)
        .run();
    rustc()
        .input("main.rs")
        .arg("-Clto=thin")
        .opt()
        .codegen_units(cg_units)
        .profile_generate(&path_prof_data_dir)
        .opt()
        .run();
    run("main");
    llvm_profdata().merge().output(&path_merged_profdata).input(path_prof_data_dir).run();
    rustc()
        .input("interesting.rs")
        .profile_use(&path_merged_profdata)
        .opt()
        .crate_type("lib,cdylib")
        .codegen_units(cg_units)
        .emit("link")
        .run();
    rustc()
        .input("main.rs")
        .arg("-Clto=thin")
        .opt()
        .codegen_units(cg_units)
        .profile_use(&path_merged_profdata)
        .emit("llvm-ir,link")
        .opt()
        .run();
    let files = shallow_find_files(".", |path| {
        has_prefix(path, "main.interesting.interesting") && has_extension(path, "ll")
    });
    assert_eq!(files.len(), 1);
    let llvm_ir = &files[0];
    llvm_filecheck().patterns("interesting.rs").stdin_buf(rfs::read(llvm_ir)).run();
}

fn main() {
    run_test(1);
    run_test(16);
}
