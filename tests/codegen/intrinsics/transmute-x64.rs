//@ compile-flags: -O -C no-prepopulate-passes
//@ only-x86_64 (it's using arch-specific types)

#![crate_type = "lib"]

use std::arch::x86_64::{__m128, __m128i, __m256i};
use std::mem::transmute;

// CHECK-LABEL: @check_sse_float_to_int(
#[no_mangle]
pub unsafe fn check_sse_float_to_int(x: __m128) -> __m128i {
    // FIXME: the MIR opt still works, but the ABI logic now introduces
    // an alloca here.
    // CHECK: alloca
    // CHECK: store <4 x float> %x, ptr %_0, align 16
    transmute(x)
}

// CHECK-LABEL: @check_sse_pair_to_avx(
#[no_mangle]
pub unsafe fn check_sse_pair_to_avx(x: (__m128i, __m128i)) -> __m256i {
    // CHECK-NOT: alloca
    // CHECK: %0 = load <4 x i64>, ptr %x, align 16
    // CHECK: store <4 x i64> %0, ptr %_0, align 32
    transmute(x)
}

// CHECK-LABEL: @check_sse_pair_from_avx(
#[no_mangle]
pub unsafe fn check_sse_pair_from_avx(x: __m256i) -> (__m128i, __m128i) {
    // CHECK-NOT: alloca
    // CHECK: %0 = load <4 x i64>, ptr %x, align 32
    // CHECK: store <4 x i64> %0, ptr %_0, align 16
    transmute(x)
}
