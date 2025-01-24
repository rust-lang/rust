//@ add-core-stubs
//@ revisions: sparc sparcv8plus sparc64
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @cc_clobber
// CHECK: call void asm sideeffect "", "~{icc},~{fcc0},~{fcc1},~{fcc2},~{fcc3}"()
#[no_mangle]
pub unsafe fn cc_clobber() {
    asm!("", options(nostack, nomem));
}

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// CHECK-LABEL: @y_clobber
// CHECK: call void asm sideeffect "", "~{y}"()
#[no_mangle]
pub unsafe fn y_clobber() {
    asm!("", out("y") _, options(nostack, nomem, preserves_flags));
}
