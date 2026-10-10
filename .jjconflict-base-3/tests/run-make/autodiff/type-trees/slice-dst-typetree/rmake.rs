//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable,NoPostopt")
        .opt_level("0")
        .arg("-Clto=fat")
        .emit("llvm-ir")
        .run();

    let ir = rfs::read("test.ll");
    llvm_filecheck().patterns("slice-dst.check").check_prefix("OSSTR").stdin_buf(&ir).run();
    llvm_filecheck().patterns("slice-dst.check").check_prefix("HEADER").stdin_buf(&ir).run();
    llvm_filecheck().patterns("slice-dst.check").check_prefix("ZST").stdin_buf(&ir).run();
}
