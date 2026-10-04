// `-Z packed_bundled_libs` is an unstable rustc flag that makes the compiler
// only require a native library and no supplementary object files to compile.
// Output files compiled with this flag should still contain all expected symbols -
// that is what this test checks.
// See https://github.com/rust-lang/rust/pull/100101

//@ ignore-cross-compile
// Reason: cross-compilation fails to export native symbols

use run_make_support::{
    bin_name, build_native_static_lib, cwd, filename_contains, is_windows_msvc, llvm_ar, llvm_nm,
    rfs, rust_lib_name, rustc, shallow_find_files,
};

fn main() {
    build_native_static_lib("native_dep_1");
    build_native_static_lib("native_dep_2");
    build_native_static_lib("native_dep_3");
    rustc().input("rust_dep_up.rs").crate_type("rlib").arg("-Zpacked_bundled_libs").run();
    llvm_nm()
        .input(rust_lib_name("rust_dep_up"))
        .run()
        .assert_stdout_contains_regex("U.*native_f2");
    llvm_nm()
        .input(rust_lib_name("rust_dep_up"))
        .run()
        .assert_stdout_contains_regex("U.*native_f3");
    llvm_nm()
        .input(rust_lib_name("rust_dep_up"))
        .run()
        .assert_stdout_contains_regex("T.*rust_dep_up");
    llvm_ar()
        .table_of_contents()
        .arg(rust_lib_name("rust_dep_up"))
        .run()
        .assert_stdout_contains("native_dep_2");
    llvm_ar()
        .table_of_contents()
        .arg(rust_lib_name("rust_dep_up"))
        .run()
        .assert_stdout_contains("native_dep_3");
    rustc()
        .input("rust_dep_local.rs")
        .extern_("rlib", rust_lib_name("rust_dep_up"))
        .arg("-Zpacked_bundled_libs")
        .crate_type("rlib")
        .run();
    llvm_nm()
        .input(rust_lib_name("rust_dep_local"))
        .run()
        .assert_stdout_contains_regex("U.*native_f1");
    llvm_nm()
        .input(rust_lib_name("rust_dep_local"))
        .run()
        .assert_stdout_contains_regex("T.*rust_dep_local");
    llvm_ar()
        .table_of_contents()
        .arg(rust_lib_name("rust_dep_local"))
        .run()
        .assert_stdout_contains("native_dep_1");

    // Ensure the compiler will not use files it should not know about.
    for file in shallow_find_files(cwd(), |path| filename_contains(path, "native_dep_")) {
        rfs::remove_file(file);
    }

    rustc()
        .input("main.rs")
        .extern_("lib", rust_lib_name("rust_dep_local"))
        .output(bin_name("main"))
        .arg("-Zpacked_bundled_libs")
        .print("link-args")
        .run()
        .assert_stdout_contains_regex("native_dep_1.*native_dep_2.*native_dep_3");

    // The binary "main" will not contain any symbols on MSVC.
    if !is_windows_msvc() {
        llvm_nm().input(bin_name("main")).run().assert_stdout_contains_regex("T.*native_f1");
        llvm_nm().input(bin_name("main")).run().assert_stdout_contains_regex("T.*native_f2");
        llvm_nm().input(bin_name("main")).run().assert_stdout_contains_regex("T.*native_f3");
        llvm_nm().input(bin_name("main")).run().assert_stdout_contains_regex("T.*rust_dep_local");
        llvm_nm().input(bin_name("main")).run().assert_stdout_contains_regex("T.*rust_dep_up");
    }
}
