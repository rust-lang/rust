//@ ignore-nvptx64
//@ ignore-wasm
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{
    cc, dynamic_lib_name, is_darwin, is_windows, llvm_ar, llvm_nm, llvm_readobj, rfs, rustc,
    static_lib_name,
};

fn main() {
    cc().input("foo.c").arg("-c").arg("-fno-lto").out_exe("foo.o").run();
    llvm_ar().obj_to_ar().output_input(&static_lib_name("foo"), "foo.o").run();

    rustc().input("foo.rs").arg("-lstatic=foo").crate_type("cdylib").run();

    if is_darwin() {
        llvm_nm().input(dynamic_lib_name("foo")).run().assert_stdout_not_contains("T _my_function");
    } else if is_windows() {
        llvm_readobj()
            .arg("--coff-exports")
            .input(dynamic_lib_name("foo"))
            .run()
            .assert_stdout_not_contains("my_function");
    } else {
        llvm_nm().input(dynamic_lib_name("foo")).run().assert_stdout_not_contains("T my_function");
    }

    rfs::remove_file(dynamic_lib_name("foo"));

    rustc().input("foo_export.rs").arg("-lstatic:+export-symbols=foo").crate_type("cdylib").run();

    if is_darwin() {
        llvm_nm()
            .input(dynamic_lib_name("foo_export"))
            .run()
            .assert_stdout_contains("T _my_function");
    } else if is_windows() {
        llvm_readobj()
            .arg("--coff-exports")
            .input(dynamic_lib_name("foo_export"))
            .run()
            .assert_stdout_contains("my_function");
    } else {
        llvm_nm()
            .input(dynamic_lib_name("foo_export"))
            .run()
            .assert_stdout_contains("T my_function");
    }
}
