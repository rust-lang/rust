use std::fs;

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // Compile the Rust file with the required flags, capturing both stdout and stderr
    let output = rustc()
        .input("struct.rs")
        .arg("-Zautodiff=Enable,PrintTAFn=callee")
        .arg("-Zautodiff=NoPostopt")
        .arg("-Copt-level=3")
        .arg("-Clto=fat")
        .arg("-g")
        .run();

    let stdout = output.stdout_utf8();
    let stderr = output.stderr_utf8();

    // Write the outputs to files
    fs::write("struct.stdout", stdout).unwrap();
    fs::write("struct.stderr", stderr).unwrap();

    // Run FileCheck on the stdout using the check file
    llvm_filecheck().patterns("struct.check").stdin_buf(rfs::read("struct.stdout")).run();
}
