//@ add-minicore
//@ assembly-output: emit-asm
//
//@ compile-flags: -Copt-level=3 --target=msp430-none-elf
//@ needs-llvm-components: msp430
#![feature(c_variadic_experimental_arch, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::ffi::VaList;
use minicore::*;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // CHECK: mov #1, r15
    // CHECK-NEXT: add 0(r12), r15
    // CHECK-NEXT: and #-2, r15
    // CHECK-NEXT: mov r15, r13
    // CHECK-NEXT: add #8, r13
    // CHECK-NEXT: mov r13, 0(r12)
    // CHECK-NEXT: mov 0(r15), r12
    // CHECK-NEXT: mov 2(r15), r13
    // CHECK-NEXT: mov 4(r15), r14
    // CHECK-NEXT: mov 6(r15), r15
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i16(ap: &mut VaList<'_>) -> i16 {
    // CHECK-LABEL: read_i16
    //
    // CHECK: mov #1, r13
    // CHECK-NEXT: add 0(r12), r13
    // CHECK-NEXT: and #-2, r13
    // CHECK-NEXT: mov r13, r14
    // CHECK-NEXT: incd r14
    // CHECK-NEXT: mov r14, 0(r12)
    // CHECK-NEXT: mov 0(r13), r12
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // CHECK: mov #1, r13
    // CHECK-NEXT: add 0(r12), r13
    // CHECK-NEXT: and #-2, r13
    // CHECK-NEXT: mov r13, r14
    // CHECK-NEXT: add #4, r14
    // CHECK-NEXT: mov r14, 0(r12)
    // CHECK-NEXT: mov 0(r13), r12
    // CHECK-NEXT: mov 2(r13), r13
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // CHECK: mov #1, r15
    // CHECK-NEXT: add 0(r12), r15
    // CHECK-NEXT: and #-2, r15
    // CHECK-NEXT: mov r15, r13
    // CHECK-NEXT: add #8, r13
    // CHECK-NEXT: mov r13, 0(r12)
    // CHECK-NEXT: mov 0(r15), r12
    // CHECK-NEXT: mov 2(r15), r13
    // CHECK-NEXT: mov 4(r15), r14
    // CHECK-NEXT: mov 6(r15), r15
    // CHECK-NEXT: ret
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK: read_ptr = read_i16
    ap.next_arg()
}
