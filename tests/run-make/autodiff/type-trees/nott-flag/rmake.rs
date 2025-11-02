//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // Test with NoTT flag - should not generate TypeTree metadata
    rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable,NoTT")
        .emit("llvm-ir")
        .arg("-o")
        .arg("nott.ll")
        .run();

    // Test without NoTT flag - should generate TypeTree metadata
    rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable")
        .emit("llvm-ir")
        .arg("-o")
        .arg("with_tt.ll")
        .run();

    // Verify NoTT version does NOT have enzyme_type attributes
    llvm_filecheck().patterns("nott.check").stdin_buf(rfs::read("nott.ll")).run();

    // Verify TypeTree version DOES have enzyme_type attributes
    llvm_filecheck().patterns("with_tt.check").stdin_buf(rfs::read("with_tt.ll")).run();
}
