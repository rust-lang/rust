//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // Compile with TypeTree enabled and emit LLVM IR
    rustc().input("test.rs").arg("-Zautodiff=Enable").emit("llvm-ir").run();

    // Check that f128 TypeTree metadata is correctly generated
    llvm_filecheck().patterns("f128.check").stdin_buf(rfs::read("test.ll")).run();
}
