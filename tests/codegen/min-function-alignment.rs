//@ revisions: align16 align1024
//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0 -Clink-dead-code
//@ [align16] compile-flags: -Zmin-function-alignment=16
//@ [align1024] compile-flags: -Zmin-function-alignment=1024
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)

#![crate_type = "lib"]
// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

// Functions without explicit alignment use the global minimum.
//
// NOTE: this function deliberately has zero (0) attributes! That is to make sure that
// `-Zmin-function-alignment` is applied regardless of whether attributes are used.
//
// CHECK-LABEL: no_explicit_align
// align16: align 16
// align1024: align 1024
pub fn no_explicit_align() {}

// CHECK-LABEL: @lower_align
// align16: align 16
// align1024: align 1024
#[no_mangle]
#[rustc_align(8)]
pub fn lower_align() {}

// the higher value of min-function-alignment and the align attribute wins out
//
// CHECK-LABEL: @higher_align
// align16: align 32
// align1024: align 1024
#[no_mangle]
#[rustc_align(32)]
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
