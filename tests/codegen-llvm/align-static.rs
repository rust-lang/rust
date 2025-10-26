//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
#![feature(static_align)]

// CHECK: @STATIC_ALIGN =
// CHECK-SAME: align 16
#[no_mangle]
#[rustc_align_static(16)]
pub static STATIC_ALIGN: u64 = 0;

// CHECK: @ALIGN_SPECIFIED_TWICE_1 =
// CHECK-SAME: align 64
#[no_mangle]
#[rustc_align_static(32)]
#[rustc_align_static(64)]
pub static ALIGN_SPECIFIED_TWICE_1: u64 = 0;

// CHECK: @ALIGN_SPECIFIED_TWICE_2 =
// CHECK-SAME: align 128
#[no_mangle]
#[rustc_align_static(128)]
#[rustc_align_static(32)]
pub static ALIGN_SPECIFIED_TWICE_2: u64 = 0;

// CHECK: @ALIGN_SPECIFIED_TWICE_3 =
// CHECK-SAME: align 256
#[no_mangle]
#[rustc_align_static(32)]
#[rustc_align_static(256)]
pub static ALIGN_SPECIFIED_TWICE_3: u64 = 0;
