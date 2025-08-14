// This test case makes sure that native libraries are linked with appropriate semantics
// when the `[+-]bundle,[+-]whole-archive` modifiers are applied to them.
// The test works by checking that the resulting executables produce the expected output,
// part of which is emitted by otherwise unreferenced C code. If +whole-archive didn't work
// that code would never make it into the final executable and we'd thus be missing some
// of the output.
// See https://github.com/rust-lang/rust/issues/88085

//@ ignore-cross-compile
// Reason: compiling C++ code does not work well when cross-compiling
// plus, the compiled binary is executed

use run_make_support::{cxx, is_windows_msvc, llvm_ar, run, run_with_args, rustc, static_lib_name};

fn main() {
    let mut cxx = cxx();
    if is_windows_msvc() {
        cxx.arg("-EHs");
    }
    cxx.input("c_static_lib_with_constructor.cpp")
        .arg("-c")
        .out_exe("libc_static_lib_with_constructor")
        .run();

    let mut llvm_ar = llvm_ar();
    llvm_ar.obj_to_ar();
    if is_windows_msvc() {
        llvm_ar
            .output_input(
                static_lib_name("c_static_lib_with_constructor"),
                "libc_static_lib_with_constructor.obj",
            )
            .run();
    } else {
        llvm_ar
            .output_input(
                static_lib_name("c_static_lib_with_constructor"),
                "libc_static_lib_with_constructor",
            )
            .run();
    }

    // Native lib linked directly into executable
    rustc()
        .input("directly_linked.rs")
        .arg("-lstatic:+whole-archive=c_static_lib_with_constructor")
        .run();

    // Native lib linked into test executable, +whole-archive
    rustc()
        .input("directly_linked_test_plus_whole_archive.rs")
        .arg("--test")
        .arg("-lstatic:+whole-archive=c_static_lib_with_constructor")
        .run();

    // Native lib linked into test executable, -whole-archive
    rustc()
        .input("directly_linked_test_minus_whole_archive.rs")
        .arg("--test")
        .arg("-lstatic:-whole-archive=c_static_lib_with_constructor")
        .run();

    // Native lib linked into rlib with via commandline
    rustc()
        .input("rlib_with_cmdline_native_lib.rs")
        .crate_type("rlib")
        .arg("-lstatic:-bundle,+whole-archive=c_static_lib_with_constructor")
        .run();
    // Native lib linked into RLIB via `-l static:-bundle,+whole-archive`
    // RLIB linked into executable
    rustc().input("indirectly_linked.rs").run();

    // Native lib linked into rlib via `#[link()]` attribute on extern block.
    rustc().input("native_lib_in_src.rs").crate_type("rlib").run();
    // Native lib linked into RLIB via #[link] attribute, RLIB linked into executable
    rustc().input("indirectly_linked_via_attr.rs").run();

    run("directly_linked").assert_stdout_contains("static-initializer.directly_linked.");
    run_with_args("directly_linked_test_plus_whole_archive", &["--nocapture"])
        .assert_stdout_contains("static-initializer.");
    run_with_args("directly_linked_test_minus_whole_archive", &["--nocapture"])
        .assert_stdout_not_contains("static-initializer.");
    run("indirectly_linked").assert_stdout_contains("static-initializer.indirectly_linked.");
    run("indirectly_linked_via_attr")
        .assert_stdout_contains("static-initializer.native_lib_in_src.");
}
