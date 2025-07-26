// When we generate the import library for a dylib or bin crate, we should generate it
// for the symbols both for the current crate and all upstream crates. This allows for
// using the link kind `raw-dylib` inside inline functions successfully. This test checks
// that the import symbols in the object files match this convention, and that execution
// of the binary results in all function names exported successfully.
// See https://github.com/rust-lang/rust/pull/102988

//@ only-windows

use run_make_support::{cc, diff, is_windows_msvc, llvm_objdump, run, rustc};

fn main() {
    rustc()
        .crate_type("dylib")
        .crate_name("raw_dylib_test")
        .input("lib.rs")
        .arg("-Cprefer-dynamic")
        .run();
    rustc()
        .crate_type("dylib")
        .crate_name("raw_dylib_test_wrapper")
        .input("lib_wrapper.rs")
        .arg("-Cprefer-dynamic")
        .run();
    rustc().crate_type("bin").input("driver.rs").arg("-Cprefer-dynamic").run();
    llvm_objdump()
        .arg("--private-headers")
        .input("driver.exe")
        .run()
        // Make sure we don't find an import to the functions we expect to be inlined.
        .assert_stdout_not_contains("inline_library_function")
        // Make sure we do find an import to the functions we expect to be imported.
        .assert_stdout_contains("library_function");
    if is_windows_msvc() {
        cc().arg("-c").out_exe("extern_1").input("extern_1.c").run();
        cc().arg("-c").out_exe("extern_2").input("extern_2.c").run();
        cc().input("extern_1.obj")
            .arg("-link")
            .arg("-dll")
            .arg("-out:extern_1.dll")
            .arg("-noimplib")
            .run();
        cc().input("extern_2.obj")
            .arg("-link")
            .arg("-dll")
            .arg("-out:extern_2.dll")
            .arg("-noimplib")
            .run();
    } else {
        cc().arg("-v").arg("-c").out_exe("extern_1").input("extern_1.c").run();
        cc().arg("-v").arg("-c").out_exe("extern_2").input("extern_2.c").run();
        cc().input("extern_1").out_exe("extern_1.dll").arg("-shared").run();
        cc().input("extern_2").out_exe("extern_2.dll").arg("-shared").run();
    }
    let out = run("driver").stdout_utf8();
    diff()
        .expected_file("output.txt")
        .actual_text("actual_output", out)
        .normalize(r#"\r"#, "")
        .run();
}
