//@ ignore-cross-compile

use run_make_support::fs_wrapper::read;
use run_make_support::{assert_contains, run, rustc};

fn main() {
    // The tests must pass...
    rustc().input("main.rs").run();
    run("main");

    // ... and the loads/stores must not be optimized out.
    rustc().input("main.rs").emit("llvm-ir").run();

    let raw_llvm_ir = read("main.ll");
    let llvm_ir = String::from_utf8_lossy(&raw_llvm_ir);
    assert_contains(&llvm_ir, "load volatile");
    assert_contains(&llvm_ir, "store volatile");
}
