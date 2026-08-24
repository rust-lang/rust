// Make sure that no 0-sized padding is inserted in structs and that
// structs are represented as expected by Neon intrinsics in LLVM.
// See #87254.
//@ add-minicore
//@ compile-flags: -Cno-prepopulate-passes --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(abi_unadjusted, link_llvm_intrinsics, no_core)]
#![no_core]

extern crate minicore;

use minicore::simd::i16x4;

#[repr(C)]
struct int16x4x2_t(i16x4, i16x4);

unsafe extern "unadjusted" {
    #[link_name = "llvm.aarch64.neon.ld1x2.v4i16.p0"]
    fn vld1_s16_x2(t: *const i16) -> int16x4x2_t;
}

#[no_mangle]
unsafe extern "C" fn returns_int16x4x2_t(a: *const i16) -> int16x4x2_t {
    // CHECK: call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld1x2.v4i16.p0
    vld1_s16_x2(a)
}
