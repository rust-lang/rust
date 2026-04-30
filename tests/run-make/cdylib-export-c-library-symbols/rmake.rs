//@ ignore-nvptx64
//@ ignore-wasm
//@ ignore-cross-compile
// FIXME:The symbol mangle rules are slightly different in Windows(32-bit) and Apple.
// Need to be resolved.
//@ ignore-windows
//@ ignore-apple
// Reason: the compiled binary is executed

use run_make_support::{cc, dynamic_lib_name, is_darwin, llvm_ar, llvm_nm, rustc, static_lib_name};

fn main() {
    // Compile C code without LTO
    cc().input("foo.c").arg("-c").arg("-fno-lto").out_exe("foo.o").run();
    llvm_ar().obj_to_ar().output_input(&static_lib_name("foo"), "foo.o").run();

    rustc().input("foo.rs").arg("-lstatic=foo").crate_type("cdylib").run();

    let out = llvm_nm()
        .input(dynamic_lib_name("foo"))
        .run()
        .assert_stdout_not_contains_regex("T *my_function");

    rustc().input("foo_export.rs").arg("-lstatic:+export-symbols=foo").crate_type("cdylib").run();

    if is_darwin() {
        let out = llvm_nm()
            .input(dynamic_lib_name("foo_export"))
            .run()
            .assert_stdout_contains("T _my_function");
    } else {
        let out = llvm_nm()
            .input(dynamic_lib_name("foo_export"))
            .run()
            .assert_stdout_contains("T my_function");
    }
}
