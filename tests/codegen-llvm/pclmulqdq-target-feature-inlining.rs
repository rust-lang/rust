//@ only-x86_64
//@ compile-flags: -C opt-level=3

//! Regression test for https://github.com/rust-lang/rust/issues/139029
//!
//! The pclmulqdq intrinsics should inline into functions with the required target features.

#![crate_type = "lib"]

use std::arch::x86_64 as arch;

// CHECK-LABEL: @reduce128_caller
// CHECK-NEXT: start
// CHECK-COUNT-3: load
// CHECK-NEXT: call <2 x i64> @llvm.x86.pclmulqdq
// CHECK-NEXT: call <2 x i64> @llvm.x86.pclmulqdq
#[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
#[no_mangle]
pub unsafe fn reduce128_caller(
    a: arch::__m128i,
    b: arch::__m128i,
    keys: arch::__m128i,
) -> arch::__m128i {
    reduce128(a, b, keys)
}

unsafe fn reduce128(a: arch::__m128i, b: arch::__m128i, keys: arch::__m128i) -> arch::__m128i {
    let t1 = arch::_mm_clmulepi64_si128(a, keys, 0x00);
    let t2 = arch::_mm_clmulepi64_si128(a, keys, 0x11);
    arch::_mm_xor_si128(arch::_mm_xor_si128(b, t1), t2)
}
