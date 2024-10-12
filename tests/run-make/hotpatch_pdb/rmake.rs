// Check if hotpatch flag is present in the Codeview.
// This is need so linkers actually pad functions when given the functionpadmin arg.

//@ revisions: x32 x64 aarch64
//@[x32] only-x86
//@[x64] only-x86_64
//@[aarch64] only-aarch64

// Reason: Hotpatch is only implemented for x86 and aarch64

use run_make_support::{llvm, rustc};

fn main() {
    let output = rustc()
        .input("main.rs")
        .arg("-g")
        .arg("-Zhotpatch")
        .crate_name("hotpatch_pdb")
        .crate_type("bin")
        .run();

    let pdbutil_output = llvm::llvm_pdbutil()
        .arg("dump")
        .arg("-symbols")
        .input("hotpatch_pdb.pdb")
        .run()
        .stdout_utf8();

    llvm::llvm_filecheck().patterns("main.rs").stdin_buf(&pdbutil_output).run();
}
