// Compile a library with lto=fat, then compile a binary with lto=thin
// and check that lto is applied with the library.
// The goal is to mimic the standard library being build with lto=fat
// and allowing users to build with lto=thin.

//@ only-x86_64-unknown-linux-gnu

use run_make_support::{dynamic_lib_name, llvm_objdump, rustc};

fn main() {
    rustc().input("lib.rs").opt_level("3").lto("fat").run();
    rustc().input("main.rs").panic("abort").opt_level("3").lto("thin").run();

    llvm_objdump()
        .input(dynamic_lib_name("main"))
        .arg("--disassemble-symbols=bar")
        .run()
        // The called function should be inlined.
        // Check that we have a ret (to detect tail
        // calls with a jmp) and no call.
        .assert_stdout_contains("bar")
        .assert_stdout_contains("ret")
        .assert_stdout_not_contains("foo")
        .assert_stdout_not_contains("call");
}
