// Check that -C lto=fat with -C linker-plugin-lto actually works and can inline functions.
// A library is created from LLVM IR, defining a single function. Then a dylib is compiled,
// linking to the library and calling the function from the library.
// The function from the library should end up inlined and disappear from the output.

//@ only-x86_64-unknown-linux-gnu
//@ needs-rust-lld

use run_make_support::{dynamic_lib_name, llvm_as, llvm_objdump, rustc};

fn main() {
    llvm_as().input("ir.ll").run();
    rustc()
        .input("main.rs")
        .opt_level("3")
        .lto("fat")
        .linker_plugin_lto("on")
        .link_arg("ir.bc")
        .arg("-Zunstable-options")
        .arg("-Clinker-features=+lld")
        .run();

    llvm_objdump()
        .input(dynamic_lib_name("main"))
        .arg("--disassemble-symbols=rs_foo")
        .run()
        // The called function should be inlined.
        // Check that we have a ret (to detect tail
        // calls with a jmp) and no call.
        .assert_stdout_contains("foo")
        .assert_stdout_contains("ret")
        .assert_stdout_not_contains("call");
}
