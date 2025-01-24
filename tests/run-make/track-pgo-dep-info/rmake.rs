// Emitting dep-info files used to not have any mention of PGO profiles used
// in compilation, which meant these profiles could be changed without consequence.
// After changing this in #100801, this test checks that the profile data is successfully
// included in dep-info emit files.
// See https://github.com/rust-lang/rust/pull/100801

//@ ignore-cross-compile
// Reason: the binary is executed
//@ needs-profiler-runtime

use run_make_support::{llvm_profdata, rfs, run, rustc};

fn main() {
    // Generate the profile-guided-optimization (PGO) profiles
    rustc().profile_generate("profiles").input("main.rs").run();
    // Merge the profiles
    run("main");
    llvm_profdata().merge().output("merged.profdata").input("profiles").run();
    // Use the profiles in compilation
    rustc().profile_use("merged.profdata").emit("dep-info").input("main.rs").run();
    // Check that the profile file is in the dep-info emit file
    assert!(rfs::read_to_string("main.d").contains("merged.profdata"));
}
