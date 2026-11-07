// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ needs-sanitizer-support
//@ needs-sanitizer-cfi
//@ ignore-backends: gcc

use run_make_support::{llvm_bcanalyzer, llvm_nm, rustc};

fn main() {
    rustc()
        .arg("-Clto=thin")
        .arg("-Clinker-plugin-lto")
        .arg("-Ccodegen-units=1")
        .arg("-Ctarget-feature=-crt-static")
        .arg("-Cunsafe-allow-abi-mismatch=sanitizer")
        .arg("-Cdebuginfo=2")
        .arg("-Zsanitizer=cfi")
        // -Zunstable-options is required for thin-link-bitcode
        .arg("-Zunstable-options")
        .crate_type("lib")
        .emit("obj,thin-link-bitcode")
        .input("program.rs")
        .run();
    check_thin_link_bitcode("program.o", "program.indexing.o");
}

fn check_thin_link_bitcode(obj: &str, thin_link: &str) {
    // Making sure that the .o file is not stripped from debug info while the .indexing.o is
    // stripped.
    let obj_dump = llvm_bcanalyzer()
        .arg("-dump")
        .input(obj)
        .run()
        .assert_stdout_contains("COMPILE_UNIT")
        .stdout_utf8();
    let thin_link_dump = llvm_bcanalyzer()
        .arg("-dump")
        .input(thin_link)
        .run()
        .assert_stdout_not_contains("COMPILE_UNIT")
        .stdout_utf8();

    // Making sure that the module hash is identical between the .o and the .indexing.o file.
    let obj_hash = obj_dump.lines().find(|line| line.trim().starts_with("<HASH ")).unwrap();
    let thin_link_hash =
        thin_link_dump.lines().find(|line| line.trim().starts_with("<HASH ")).unwrap();
    assert_eq!(obj_hash, thin_link_hash);
}
