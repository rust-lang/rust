// Check if the pdb file contains an S_OBJNAME entry with the name of the .o file

// This is because it used to be missing in #96475.
// See https://github.com/rust-lang/rust/pull/115704

//@ only-windows-msvc
// Reason: pdb files are unique to this architecture

use run_make_support::{llvm, rustc};

fn main() {
    rustc().input("main.rs").arg("-g").crate_name("my_great_crate_name").crate_type("bin").run();

    let pdbutil_result = llvm::llvm_pdbutil()
        .arg("dump")
        .arg("-symbols")
        .input("my_great_crate_name.pdb")
        .run()
        .assert_stdout_contains_regex("S_OBJNAME.+my_great_crate_name.*\\.o");
}
