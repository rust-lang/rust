//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

// This test passes in release mode. If we run it in Debug mode and don't lower any MIR info to
// LLVM TypeTrees, then it fails on deducing the type of a memcpy. If we lower info it still fails,
// but at a later location based on an extractvalue call. We will fix this in a future PR.

fn main() {
    rustc()
        .input("window.rs")
        .arg("-Zautodiff=Enable,NoTT")
        .arg("-Clto=fat")
        .run_fail()
        .assert_stderr_contains("Enzyme: Cannot deduce type of copy");
    rustc().input("window.rs").arg("-Zautodiff=Enable,NoTT").arg("-Clto=fat").arg("-O").run();
    rustc().input("window.rs").arg("-Zautodiff=Enable").arg("-Clto=fat").emit("llvm-ir").run();
    llvm_filecheck().patterns("window.check").stdin_buf(rfs::read("window.ll")).run();
}
