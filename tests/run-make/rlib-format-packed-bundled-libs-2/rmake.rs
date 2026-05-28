//@ ignore-cross-compile

// `-Z packed_bundled_libs` is an unstable rustc flag that makes the compiler
// only require a native library and no supplementary object files to compile.
// This test simply checks that this flag can be passed alongside verbatim syntax
// in rustc flags without a compilation failure or the removal of expected symbols.
// See https://github.com/rust-lang/rust/pull/100101

use run_make_support::{llvm_ar, llvm_nm, rfs, rust_lib_name, rustc};

fn main() {
    // Build a strangely named dependency.
    rustc().input("native_dep.rs").crate_type("staticlib").output("native_dep.ext").run();

    rustc().input("rust_dep.rs").crate_type("rlib").arg("-Zpacked_bundled_libs").run();
    llvm_nm().input(rust_lib_name("rust_dep")).run().assert_stdout_contains_regex("U.*native_f1");
    llvm_ar()
        .arg("t")
        .arg(rust_lib_name("rust_dep"))
        .run()
        .assert_stdout_contains("native_dep.ext");

    // Ensure the compiler does not use files it should not be aware of.
    rfs::remove_file("native_dep.ext");
    rustc()
        .input("main.rs")
        .extern_("rust_dep", rust_lib_name("rust_dep"))
        .arg("-Zpacked_bundled_libs")
        .run();
}
