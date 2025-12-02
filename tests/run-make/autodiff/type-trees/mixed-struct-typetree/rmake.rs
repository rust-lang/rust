//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("0")
        .emit("llvm-ir")
        .run();

    llvm_filecheck().patterns("mixed.check").stdin_buf(rfs::read("test.ll")).run();
}
