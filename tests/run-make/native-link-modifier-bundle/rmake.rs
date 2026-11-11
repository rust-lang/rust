// This test exercises the `bundle` link argument, which can be turned on or off.

// When building a rlib or staticlib, +bundle means that all object files from the native static
// library will be added to the rlib or staticlib archive, and then used from it during linking of
// the final binary.

// When building a rlib -bundle means that the native static library is registered as a dependency
// of that rlib "by name", and object files from it are included only during linking of the final
// binary, the file search by that name is also performed during final linking.
// When building a staticlib -bundle means that the native static library is simply not included
// into the archive and some higher level build system will need to add it later during linking of
// the final binary.

// This modifier has no effect when building other targets like executables or dynamic libraries.

// The default for this modifier is +bundle.
// See https://github.com/rust-lang/rust/pull/95818

//@ ignore-cross-compile
// Reason: cross-compilation fails to export native symbols

use run_make_support::{
    build_native_static_lib, dynamic_lib_name, is_windows_msvc, llvm_nm, rust_lib_name, rustc,
    static_lib_name,
};

fn main() {
    build_native_static_lib("native-staticlib");
    // Build a staticlib and a rlib, the `native_func` symbol will be bundled into them
    rustc().input("bundled.rs").crate_type("staticlib").crate_type("rlib").run();
    llvm_nm()
        .input(static_lib_name("bundled"))
        .run()
        .assert_stdout_contains_regex("T _*native_func");
    llvm_nm()
        .input(static_lib_name("bundled"))
        .run()
        .assert_stdout_contains_regex("U _*native_func");
    llvm_nm().input(rust_lib_name("bundled")).run().assert_stdout_contains_regex("T _*native_func");
    llvm_nm().input(rust_lib_name("bundled")).run().assert_stdout_contains_regex("U _*native_func");

    // Build a staticlib and a rlib, the `native_func` symbol will not be bundled into it
    build_native_static_lib("native-staticlib");
    rustc().input("non-bundled.rs").crate_type("staticlib").crate_type("rlib").run();
    llvm_nm()
        .input(static_lib_name("non_bundled"))
        .run()
        .assert_stdout_not_contains_regex("T _*native_func");
    llvm_nm()
        .input(static_lib_name("non_bundled"))
        .run()
        .assert_stdout_contains_regex("U _*native_func");
    llvm_nm()
        .input(rust_lib_name("non_bundled"))
        .run()
        .assert_stdout_not_contains_regex("T _*native_func");
    llvm_nm()
        .input(rust_lib_name("non_bundled"))
        .run()
        .assert_stdout_contains_regex("U _*native_func");

    // This part of the test does not function on Windows MSVC - no symbols are printed.
    if !is_windows_msvc() {
        // Build a cdylib, `native-staticlib` will not appear on the linker line because it was
        // bundled previously. The cdylib will contain the `native_func` symbol in the end.
        rustc()
            .input("cdylib-bundled.rs")
            .crate_type("cdylib")
            .print("link-args")
            .run()
            .assert_stdout_not_contains(r#"-l[" ]*native-staticlib"#);
        llvm_nm()
            .input(dynamic_lib_name("cdylib_bundled"))
            .run()
            .assert_stdout_contains_regex("[Tt] _*native_func");

        // Build a cdylib, `native-staticlib` will appear on the linker line because it was not
        // bundled previously. The cdylib will contain the `native_func` symbol in the end
        rustc()
            .input("cdylib-non-bundled.rs")
            .crate_type("cdylib")
            .print("link-args")
            .run()
            .assert_stdout_contains_regex(r#"-l[" ]*native-staticlib"#);
        llvm_nm()
            .input(dynamic_lib_name("cdylib_non_bundled"))
            .run()
            .assert_stdout_contains_regex("[Tt] _*native_func");
    }
}
