// This test makes sure that cross-language inlining can be used in conjunction
// with profile-guided optimization. The test only tests that the whole workflow
// can be executed without anything crashing. It does not test whether PGO or
// xLTO have any specific effect on the generated code.
// See https://github.com/rust-lang/rust/pull/61036

//@ needs-force-clang-based-tests
// NOTE(#126180): This test would only run on `x86_64-gnu-debug`, because that CI job sets
// RUSTBUILD_FORCE_CLANG_BASED_TESTS and only runs tests which contain "clang" in their
// name.

//@ needs-profiler-runtime
// FIXME(Oneirical): Except that due to the reliance on llvm-profdata, this test
// never runs, because `x86_64-gnu-debug` does not have the `profiler_builtins` crate.

//FIXME(Oneirical): There was a strange workaround for MSVC on this test
// which added -C panic=abort to every RUSTC call. It was justified as follows:

// "LLVM doesn't support instrumenting binaries that use SEH:
// https://bugs.llvm.org/show_bug.cgi?id=41279
// Things work fine with -Cpanic=abort though."

// This isn't very pertinent, however, as the test does not get run on any
// MSVC platforms.

use run_make_support::{
    clang, env_var, has_extension, has_prefix, llvm_ar, llvm_profdata, rfs, run, rustc,
    shallow_find_files, static_lib_name,
};

fn main() {
    rustc()
        .linker_plugin_lto("on")
        .output(static_lib_name("rustlib-xlto"))
        .opt_level("3")
        .codegen_units(1)
        .input("rustlib.rs")
        .arg("-Cprofile-generate=cpp-profdata")
        .run();
    clang()
        .lto("thin")
        .arg("-fprofile-generate=cpp-profdata")
        .use_ld("lld")
        .arg("-lrustlib-xlto")
        .out_exe("cmain")
        .input("cmain.c")
        .arg("-O3")
        .run();
    run("cmain");
    // Postprocess the profiling data so it can be used by the compiler
    let profraw_files = shallow_find_files("cpp-profdata", |path| {
        has_prefix(path, "default") && has_extension(path, "profraw")
    });
    let profraw_file = profraw_files.get(0).unwrap();
    llvm_profdata().merge().output("cpp-profdata/merged.profdata").input(profraw_file).run();
    rustc()
        .linker_plugin_lto("on")
        .profile_use("cpp-profdata/merged.profdata")
        .output(static_lib_name("rustlib-xlto"))
        .opt_level("3")
        .codegen_units(1)
        .input("rustlib.rs")
        .run();
    clang()
        .lto("thin")
        .arg("-fprofile-use=cpp-profdata/merged.profdata")
        .use_ld("lld")
        .arg("-lrustlib-xlto")
        .out_exe("cmain")
        .input("cmain.c")
        .arg("-O3")
        .run();

    clang()
        .input("clib.c")
        .arg("-fprofile-generate=rs-profdata")
        .lto("thin")
        .arg("-c")
        .out_exe("clib.o")
        .arg("-O3")
        .run();
    llvm_ar().obj_to_ar().output_input(static_lib_name("xyz"), "clib.o").run();
    rustc()
        .linker_plugin_lto("on")
        .opt_level("3")
        .codegen_units(1)
        .arg("-Cprofile-generate=rs-profdata")
        .linker(&env_var("CLANG"))
        .link_arg("-fuse-ld=lld")
        .input("main.rs")
        .output("rsmain")
        .run();
    run("rsmain");
    // Postprocess the profiling data so it can be used by the compiler
    let profraw_files = shallow_find_files("rs-profdata", |path| {
        has_prefix(path, "default") && has_extension(path, "profraw")
    });
    let profraw_file = profraw_files.get(0).unwrap();
    llvm_profdata().merge().output("rs-profdata/merged.profdata").input(profraw_file).run();
    clang()
        .input("clib.c")
        .arg("-fprofile-use=rs-profdata/merged.profdata")
        .arg("-c")
        .lto("thin")
        .out_exe("clib.o")
        .arg("-O3")
        .run();
    rfs::remove_file(static_lib_name("xyz"));
    llvm_ar().obj_to_ar().output_input(static_lib_name("xyz"), "clib.o").run();
    rustc()
        .linker_plugin_lto("on")
        .opt_level("3")
        .codegen_units(1)
        .arg("-Cprofile-use=rs-profdata/merged.profdata")
        .linker(&env_var("CLANG"))
        .link_arg("-fuse-ld=lld")
        .input("main.rs")
        .output("rsmain")
        .run();
}
