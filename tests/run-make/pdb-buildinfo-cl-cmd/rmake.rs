// Check if the pdb file contains the following information in the LF_BUILDINFO:
// 1. full path to the compiler (cl)
// 2. the commandline args to compile it (cmd)
// This is because these used to be missing in #96475.
// See https://github.com/rust-lang/rust/pull/113492

//@ only-windows-msvc
// Reason: pdb files are unique to this architecture

use run_make_support::{llvm, rustc};

fn main() {
    rustc()
        .input("main.rs")
        .arg("-g")
        .crate_name("my_crate_name")
        .crate_type("bin")
        .metadata("dc9ef878b0a48666")
        .run();

    let pdbutil_result =
        llvm::llvm_pdbutil().arg("dump").arg("-ids").input("my_crate_name.pdb").run();

    llvm::llvm_filecheck().patterns("filecheck.txt").stdin_buf(pdbutil_result.stdout_utf8()).run();
}
