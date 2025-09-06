//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // This test ensures that recursive types don't cause infinite loops
    // The compiler should complete successfully due to recursion limits
    rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("0")
        .emit("llvm-ir")
        .run();

    llvm_filecheck().patterns("recursion.check").stdin_buf(rfs::read("test.ll")).run();
}
