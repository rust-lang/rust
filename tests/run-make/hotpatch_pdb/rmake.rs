// Check if hotpatch flag is present in the Codeview.
// This is need so linkers actually pad functions when given the functionpadmin arg.

use run_make_support::{llvm, rustc};

fn main() {
    // PDBs are windows only and hotpatch is only implemented for x86 and aarch64
    #[cfg(all(
        target_os = "windows",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
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
}
