//@ needs-enzyme
//@ ignore-cross-compile

use std::fs;

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // Compile the Rust file with the required flags, capturing both stdout and stderr
    let output = rustc()
        .input("ref.rs")
        .arg("-Zautodiff=Enable,PrintTAFn=callee")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("3")
        .arg("-Clto=fat")
        .arg("-g")
        .run();

    let stdout = output.stdout_utf8();
    let stderr = output.stderr_utf8();

    // Write the outputs to files
    rfs::write("ref.stdout", stdout);
    rfs::write("ref.stderr", stderr);

    // Run FileCheck on the stdout using the check file
    llvm_filecheck().patterns("ref.check").stdin_buf(rfs::read("ref.stdout")).run();
}
