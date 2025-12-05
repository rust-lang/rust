// Test that #[used] statics are included in the final dylib for proc-macros too.

//@ ignore-cross-compile
//@ ignore-windows llvm-readobj --all doesn't show local symbols on Windows
//@ needs-crate-type: proc-macro
//@ ignore-musl (FIXME: can't find `-lunwind`)

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn main() {
    rustc().input("dep.rs").run();
    rustc().input("proc_macro.rs").run();
    llvm_readobj()
        .input(dynamic_lib_name("proc_macro"))
        .arg("--all")
        .run()
        .assert_stdout_contains("VERY_IMPORTANT_SYMBOL");
}
