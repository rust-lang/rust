//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic

#![crate_type = "lib"]
#![feature(naked_functions, fn_align)]
use std::arch::naked_asm;

// CHECK: .balign 16
// CHECK-LABEL: naked_empty:
#[repr(align(16))]
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_empty() {
    // CHECK: ret
    naked_asm!("ret");
}
