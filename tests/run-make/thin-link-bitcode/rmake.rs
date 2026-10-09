//@ ignore-backends: gcc

use run_make_support::{llvm_bcanalyzer, rustc};

fn main() {
    rustc()
        .arg("-Copt-level=2")
        .arg("-Ccodegen-units=1")
        .arg("-Cdebuginfo=2")
        .arg("-Zunstable-options")
        .crate_type("lib")
        .emit("llvm-bc,thin-link-bitcode")
        .input("program.rs")
        .run();
    check_thin_link_bitcode("program.bc", "program.indexing.o");

    rustc()
        .arg("-Copt-level=2")
        .arg("-Clinker-plugin-lto")
        .arg("-Ccodegen-units=1")
        .arg("-Cdebuginfo=2")
        .arg("-Zunstable-options")
        .crate_type("lib")
        .emit("obj,thin-link-bitcode")
        .input("program.rs")
        .run();
    check_thin_link_bitcode("program.o", "program.indexing.o");
}

fn check_thin_link_bitcode(bc: &str, thin_link: &str) {
    // Verify that bitcode file contains debug info while the thin link bitcode doesn't.
    let bc_dump = llvm_bcanalyzer()
        .arg("-dump")
        .input(bc)
        .run()
        .assert_stdout_contains("COMPILE_UNIT")
        .stdout_utf8();
    let thin_link_dump = llvm_bcanalyzer()
        .arg("-dump")
        .input(thin_link)
        .run()
        .assert_stdout_not_contains("COMPILE_UNIT")
        .stdout_utf8();

    // Ensure bitcode and thin link bitcode have the same module hash to ensure they were
    // created from the same invocation of `ThinLTOBitcodeWriterPass`. The pass is terminal
    // and running it multiple times can cause module hash inconsistencies and other
    // corruptions of the bitcode.
    let bc_hash = bc_dump.lines().find(|line| line.trim().starts_with("<HASH ")).unwrap();
    let thin_link_hash =
        thin_link_dump.lines().find(|line| line.trim().starts_with("<HASH ")).unwrap();
    assert_eq!(bc_hash, thin_link_hash);
}
