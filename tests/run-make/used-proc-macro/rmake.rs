// Test that #[used] statics are included in the final dylib for proc-macros too.

//@ needs-target-std
//@ needs-crate-type: proc-macro

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
