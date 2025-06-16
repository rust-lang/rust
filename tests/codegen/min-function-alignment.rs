//@ revisions: align16 align1024
//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0
//@ [align16] compile-flags: -Zmin-function-alignment=16
//@ [align1024] compile-flags: -Zmin-function-alignment=1024

#![crate_type = "lib"]
#![feature(fn_align)]

// functions without explicit alignment use the global minimum
//
// CHECK-LABEL: @no_explicit_align
// align16: align 16
// align1024: align 1024
#[no_mangle]
pub fn no_explicit_align() {}

// CHECK-LABEL: @lower_align
// align16: align 16
// align1024: align 1024
#[no_mangle]
#[align(8)]
pub fn lower_align() {}

// the higher value of min-function-alignment and the align attribute wins out
//
// CHECK-LABEL: @higher_align
// align16: align 32
// align1024: align 1024
#[no_mangle]
#[align(32)]
pub fn higher_align() {}

// cold functions follow the same rules as other functions
//
// in GCC, the `-falign-functions` does not apply to cold functions, but
// `-Zmin-function-alignment` applies to all functions.
//
// CHECK-LABEL: @no_explicit_align_cold
// align16: align 16
// align1024: align 1024
#[no_mangle]
#[cold]
pub fn no_explicit_align_cold() {}
