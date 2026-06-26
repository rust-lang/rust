// Regression test for MSVC link.exe failing to export weak definitions from dlls.
// See https://github.com/rust-lang/rust/pull/158294

//@ only-msvc
//@ needs-rust-lld

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn test_with_linker(linker: &str) {
    rustc().input("weak.rs").linker(linker).run();

    llvm_readobj()
        .arg("--coff-exports")
        .input(dynamic_lib_name("weak"))
        .run()
        .assert_stdout_contains("Name: weak_function")
        .assert_stdout_contains("Name: WEAK_STATIC");
}

fn main() {
    test_with_linker("link");
    test_with_linker("rust-lld");
}
