// This checks that `#[used]` passes through to the linker on
// Apple targets. This is subject to change in the future.
// See https://github.com/rust-lang/rust/pull/93718

//@ only-apple

use run_make_support::{dynamic_lib_name, llvm_readobj, rustc};

fn main() {
    rustc().opt_level("3").input("dylib_used.rs").run();
    llvm_readobj()
        .input(dynamic_lib_name("dylib_used"))
        .arg("--all")
        .run()
        .assert_stdout_contains("VERY_IMPORTANT_SYMBOL");
}
