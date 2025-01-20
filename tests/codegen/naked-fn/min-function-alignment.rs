//@ compile-flags: -C no-prepopulate-passes -Copt-level=0 -Zmin-function-alignment=16
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic

#![feature(naked_functions, fn_align)]
#![crate_type = "lib"]

// functions without explicit alignment use the global minimum
//
// CHECK: .balign 16
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_no_explicit_align() {
    core::arch::naked_asm!("ret")
}

// CHECK: .balign 16
#[no_mangle]
#[repr(align(8))]
#[naked]
pub unsafe extern "C" fn naked_lower_align() {
    core::arch::naked_asm!("ret")
}

// CHECK: .balign 32
#[no_mangle]
#[repr(align(32))]
#[naked]
pub unsafe extern "C" fn naked_higher_align() {
    core::arch::naked_asm!("ret")
}

// cold functions follow the same rules as other functions
//
// in GCC, the `-falign-functions` does not apply to cold functions, but
// `-Zmin-function-alignment` applies to all functions.
//
// CHECK: .balign 16
#[no_mangle]
#[cold]
#[naked]
pub unsafe extern "C" fn no_explicit_align_cold() {
    core::arch::naked_asm!("ret")
}
