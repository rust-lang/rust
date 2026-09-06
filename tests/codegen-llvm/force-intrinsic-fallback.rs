//@ compile-flags: --crate-type=lib -C no-prepopulate-passes -Copt-level=3
//
//@ revisions: NORMAL FALLBACK
//@ [FALLBACK] compile-flags: -Zforce-intrinsic-fallback
#![feature(core_intrinsics, funnel_shifts)]

// Check the effect of `-Zforce-intrinsic-fallback`.
//
// Without the flag, the dedicated backend lowering of the intrinsic is used.
// With the flag, the fallback body is called instead.

#[no_mangle]
pub fn call_minimum_f32(x: f32, y: f32) -> f32 {
    // CHECK-LABEL: @call_minimum_f32

    // NORMAL: call float @llvm.minimum.f32
    // NORMAL-NOT: intrinsics{{.*}}minimum

    // FALLBACK-NOT: @llvm.minimum
    // FALLBACK: call {{.*}}intrinsics{{.*}}minimum
    core::intrinsics::minimum(x, y)
}

// Codegen backends can return a list of `replaced_intrinsics`, for which codegen of the fallback is
// normally skipped. `unchecked_funnel_shl` is in that list for the LLVM backend, so we test it here
// to ensure that with the flag enabled the fallback body is actually code generated and called.

#[no_mangle]
pub fn call_funnel_shl(a: u32, b: u32, shift: u32) -> u32 {
    // CHECK-LABEL: @call_funnel_shl

    // NORMAL: call i32 @llvm.fshl.i32
    // NORMAL-NOT: funnel_shl

    // FALLBACK-NOT: @llvm.fshl
    // FALLBACK: call {{.*}}funnel_shl
    unsafe { core::intrinsics::unchecked_funnel_shl(a, b, shift) }
}
