//@ add-minicore
//@ assembly-output: emit-asm
//
//@ compile-flags: -Copt-level=3 --target=m68k-unknown-linux-gnu
//@ needs-llvm-components: m68k
#![feature(c_variadic_experimental_arch, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::ffi::VaList;
use minicore::*;

// This checks `va_arg` does the same thing as GCC.

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // CHECK: move.l (4,%sp), %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: move.l %d0, %d1
    // CHECK-NEXT: add.l #8, %d1
    // CHECK-NEXT: move.l %d1, (%a0)
    // CHECK-NEXT: move.l %d0, %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: move.l (4,%a0), %d1
    // CHECK-NEXT: rts
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // CHECK: move.l (4,%sp), %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: move.l %d0, %d1
    // CHECK-NEXT: add.l #4, %d1
    // CHECK-NEXT: move.l %d1, (%a0)
    // CHECK-NEXT: move.l %d0, %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: rts
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // CHECK: move.l (4,%sp), %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: move.l %d0, %d1
    // CHECK-NEXT: add.l #8, %d1
    // CHECK-NEXT: move.l %d1, (%a0)
    // CHECK-NEXT: move.l %d0, %a0
    // CHECK-NEXT: move.l (%a0), %d0
    // CHECK-NEXT: move.l (4,%a0), %d1
    // CHECK-NEXT: rts
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK: read_ptr = read_i32
    ap.next_arg()
}
