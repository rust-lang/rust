//@ assembly-output: emit-asm
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}

// CHECK-LABEL: ttbr0_el2:
#[no_mangle]
pub fn ttbr0_el2() {
    // CHECK: //APP
    // CHECK-NEXT: msr TTBR0_EL2, x0
    // CHECK-NEXT: //NO_APP
    unsafe {
        asm!("msr ttbr0_el2, x0");
    }
}

// CHECK-LABEL: vttbr_el2:
#[no_mangle]
pub fn vttbr_el2() {
    // CHECK: //APP
    // CHECK-NEXT: msr VTTBR_EL2, x0
    // CHECK-NEXT: //NO_APP
    unsafe {
        asm!("msr vttbr_el2, x0");
    }
}
