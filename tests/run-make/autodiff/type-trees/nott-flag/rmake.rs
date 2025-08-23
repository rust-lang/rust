//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // Test with NoTT flag - should not generate TypeTree metadata
    let output_nott = rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable,NoTT,PrintTAFn=square")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("3")
        .arg("-Clto=fat")
        .arg("-g")
        .run();

    // Write output for NoTT case
    rfs::write("nott.stdout", output_nott.stdout_utf8());
    
    // Test without NoTT flag - should generate TypeTree metadata
    let output_with_tt = rustc()
        .input("test.rs")
        .arg("-Zautodiff=Enable,PrintTAFn=square")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("3")
        .arg("-Clto=fat")
        .arg("-g")
        .run();

    // Write output for TypeTree case  
    rfs::write("with_tt.stdout", output_with_tt.stdout_utf8());

    // Verify NoTT output has minimal TypeTree info
    llvm_filecheck().patterns("nott.check").stdin_buf(rfs::read("nott.stdout")).run();
    
    // Verify normal output will have TypeTree info (once implemented)
    llvm_filecheck().patterns("with_tt.check").stdin_buf(rfs::read("with_tt.stdout")).run();
}