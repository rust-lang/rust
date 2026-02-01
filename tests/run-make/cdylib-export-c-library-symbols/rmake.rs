//@ ignore-nvptx64
//@ ignore-wasm
//@ ignore-i686-pc-windows-msvc
// FIXME:The symbol mangle rules are slightly different in 32-bit Windows. Need to be resolved.
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{build_native_static_lib, cc, dynamic_lib_name, is_darwin, llvm_nm, rustc};

fn main() {
    cc().input("foo.c").arg("-c").out_exe("foo.o").run();
    build_native_static_lib("foo");

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
