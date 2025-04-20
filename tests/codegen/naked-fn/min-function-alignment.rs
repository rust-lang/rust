//@ compile-flags: -C no-prepopulate-passes -Copt-level=0 -Zmin-function-alignment=16
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic

#![feature(fn_align)]
#![crate_type = "lib"]

// functions without explicit alignment use the global minimum
//
// CHECK: .balign 16
#[no_mangle]
#[unsafe(naked)]
pub extern "C" fn naked_no_explicit_align() {
    core::arch::naked_asm!("ret")
}

// CHECK: .balign 16
#[no_mangle]
#[repr(align(8))]
#[unsafe(naked)]
pub extern "C" fn naked_lower_align() {
    core::arch::naked_asm!("ret")
}

// CHECK: .balign 32
#[no_mangle]
#[repr(align(32))]
#[unsafe(naked)]
pub extern "C" fn naked_higher_align() {
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
#[unsafe(naked)]
pub extern "C" fn no_explicit_align_cold() {
    core::arch::naked_asm!("ret")
}
