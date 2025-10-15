//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{llvm_filecheck, rfs, rustc};

fn main() {
    // First, compile to LLVM IR to check for enzyme_type attributes
    let _ir_output = rustc()
        .input("memcpy.rs")
        .arg("-Zautodiff=Enable")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("0")
        .arg("--emit=llvm-ir")
        .arg("-o")
        .arg("main.ll")
        .run();

    // Then compile with TypeTree analysis output for the existing checks
    let output = rustc()
        .input("memcpy.rs")
        .arg("-Zautodiff=Enable,PrintTAFn=test_memcpy")
        .arg("-Zautodiff=NoPostopt")
        .opt_level("3")
        .arg("-Clto=fat")
        .arg("-g")
        .run();

    let stdout = output.stdout_utf8();
    let stderr = output.stderr_utf8();
    let ir_content = rfs::read_to_string("main.ll");

    rfs::write("memcpy.stdout", &stdout);
    rfs::write("memcpy.stderr", &stderr);
    rfs::write("main.ir", &ir_content);

    llvm_filecheck().patterns("memcpy.check").stdin_buf(stdout).run();

    llvm_filecheck().patterns("memcpy-ir.check").stdin_buf(ir_content).run();
}
