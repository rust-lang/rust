//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-x86_64 (it's using arch-specific types)

#![crate_type = "lib"]

use std::arch::x86_64::{__m128, __m128i, __m256i};
use std::mem::transmute;

// CHECK-LABEL: @check_sse_pair_to_avx(
#[no_mangle]
pub unsafe fn check_sse_pair_to_avx(x: (__m128i, __m128i)) -> __m256i {
    // CHECK: start:
    // CHECK-NOT: alloca
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 32 %_0, ptr align 16 %x, i64 32, i1 false)
    // CHECK-NEXT: ret void
    transmute(x)
}

// CHECK-LABEL: @check_sse_pair_from_avx(
#[no_mangle]
pub unsafe fn check_sse_pair_from_avx(x: __m256i) -> (__m128i, __m128i) {
    // CHECK: start:
    // CHECK-NOT: alloca
    // CHECK-NEXT: %[[TEMP:.+]] = load <4 x i64>, ptr %x, align 32
    // CHECK-NEXT: store <4 x i64> %[[TEMP]], ptr %_0, align 16
    // CHECK-NEXT: ret void
    transmute(x)
}
