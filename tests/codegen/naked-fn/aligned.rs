//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic

#![crate_type = "lib"]
#![feature(naked_functions, fn_align)]
use std::arch::naked_asm;

// CHECK: Function Attrs: naked
// CHECK-NEXT: define{{.*}}void @naked_empty()
// CHECK: align 16
#[repr(align(16))]
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_empty() {
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void asm
    // CHECK-NEXT: unreachable
    naked_asm!("ret");
}
