//@ only-x86_64
//@ ignore-sgx Tests incompatible with LVI mitigations
//@ assembly-output: emit-asm
// make sure the feature is not enabled at compile-time
//@ compile-flags: -C opt-level=3 -C target-feature=-sse4.1 -C llvm-args=-x86-asm-syntax=intel

#![crate_type = "rlib"]

use std::arch::x86_64::{__m128, _mm_blend_ps};

// Use an explicit return pointer to prevent tail call optimization.
#[no_mangle]
pub unsafe fn sse41_blend_nofeature(x: __m128, y: __m128, ret: *mut __m128) {
    let f = {
        // check that _mm_blend_ps is not being inlined into the closure
        // CHECK-LABEL: {{sse41_blend_nofeature.*closure.*:}}
        // CHECK-NOT: blendps
        // CHECK: {{call .*_mm_blend_ps.*}}
        // CHECK-NOT: blendps
        // CHECK: ret
        #[inline(never)]
        |x, y, ret: *mut __m128| unsafe { *ret = _mm_blend_ps(x, y, 0b0101) }
    };
    f(x, y, ret);
}

#[no_mangle]
#[target_feature(enable = "sse4.1")]
pub fn sse41_blend_noinline(x: __m128, y: __m128) -> __m128 {
    let f = {
        // check that _mm_blend_ps is being inlined into the closure
        // CHECK-LABEL: {{sse41_blend_noinline.*closure.*:}}
        // CHECK-NOT: _mm_blend_ps
        // CHECK: blendps
        // CHECK-NOT: _mm_blend_ps
        // CHECK: ret
        #[inline(never)]
        |x, y| unsafe { _mm_blend_ps(x, y, 0b0101) }
    };
    f(x, y)
}

#[no_mangle]
#[target_feature(enable = "sse4.1")]
pub fn sse41_blend_doinline(x: __m128, y: __m128) -> __m128 {
    // check that the closure and _mm_blend_ps are being inlined into the function
    // CHECK-LABEL: sse41_blend_doinline:
    // CHECK-NOT: {{sse41_blend_doinline.*closure.*}}
    // CHECK-NOT: _mm_blend_ps
    // CHECK: blendps
    // CHECK-NOT: {{sse41_blend_doinline.*closure.*}}
    // CHECK-NOT: _mm_blend_ps
    // CHECK: ret
    let f = {
        #[inline]
        |x, y| unsafe { _mm_blend_ps(x, y, 0b0101) }
    };
    f(x, y)
}
