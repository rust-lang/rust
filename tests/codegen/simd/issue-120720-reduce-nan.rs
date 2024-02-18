// compile-flags: -C opt-level=3 -C target-cpu=cannonlake
// only-x86_64

// In a previous implementation, _mm512_reduce_add_pd did the reduction with all fast-math flags
// enabled, making it UB to reduce a vector containing a NaN.

#![crate_type = "lib"]
#![feature(stdarch_x86_avx512, avx512_target_feature)]
use std::arch::x86_64::*;

// CHECK-label: @demo(
#[no_mangle]
#[target_feature(enable = "avx512f")] // Function-level target feature mismatches inhibit inlining
pub unsafe fn demo() -> bool {
    // CHECK: %0 = tail call reassoc double @llvm.vector.reduce.fadd.v8f64(
    // CHECK: %_0.i = fcmp uno double %0, 0.000000e+00
    // CHECK: ret i1 %_0.i
    let res = unsafe {
        _mm512_reduce_add_pd(_mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f64::NAN))
    };
    res.is_nan()
}
