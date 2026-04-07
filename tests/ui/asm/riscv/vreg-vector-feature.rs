//@ add-minicore
//@ revisions: rv64gc_v rv64gc_zve32x rv64gc_zve64d
//@[rv64gc_v] compile-flags: --target riscv64gc-unknown-none-elf -C target-feature=+v
//@[rv64gc_v] needs-llvm-components: riscv
//@[rv64gc_zve32x] compile-flags: --target riscv64gc-unknown-none-elf -C target-feature=+zve32x,+zvl32b,+zicsr
//@[rv64gc_zve32x] needs-llvm-components: riscv
//@[rv64gc_zve64d] compile-flags: --target riscv64gc-unknown-none-elf -C target-feature=+zve64d
//@[rv64gc_zve64d] needs-llvm-components: riscv
//@ ignore-backends: gcc

// Verify that vreg clobbers are accepted when the appropriate vector
// target feature is enabled.  This is the positive counterpart to
// bad-reg.rs which verifies that vreg is rejected when no vector
// feature is present.

#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub unsafe fn clobber_v0() {
    // vreg clobbers should be accepted when the vector extension is active.
    asm!("", out("v0") _);
}

pub unsafe fn clobber_v31() {
    asm!("", out("v31") _);
}

pub unsafe fn clobber_vreg_class() {
    asm!("", out(vreg) _);
}
