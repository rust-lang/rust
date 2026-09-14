//@ compile-flags: -Copt-level=0 -Ctarget-feature=+sve,+sve2
//@ edition: 2021
//@ only-aarch64

#![crate_type = "lib"]
#![feature(simd_ffi)]
#![feature(stdarch_aarch64_sve)]

// Test that `vscale * size` is generated for `memcpy` of scalable vector types

use std::arch::aarch64::*;

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn svcreate2_s16_wrapper(__dst: *mut svint16x2_t, x0: *const svint16_t, x1: *const svint16_t);
    fn svcreate3_s16_wrapper(
        __dst: *mut svint16x3_t,
        x0: *const svint16_t,
        x1: *const svint16_t,
        x2: *const svint16_t,
    );
    fn svcreate4_s16_wrapper(
        __dst: *mut svint16x4_t,
        x0: *const svint16_t,
        x1: *const svint16_t,
        x2: *const svint16_t,
        x3: *const svint16_t,
    );
}

pub fn foo(x0_val: svint16_t) {
    unsafe {
        let __pred = svptrue_b16();

        let mut __c_return_value = std::mem::MaybeUninit::uninit();
        svcreate2_s16_wrapper(__c_return_value.as_mut_ptr(), &raw const x0_val, &raw const x0_val);
        let __c_return_value = __c_return_value.assume_init();
        // CHECK: call void @svcreate2_s16_wrapper(
        // CHECK-NEXT: [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
        // CHECK-NEXT: [[VSCALE_SIZE:%.*]] = mul i64 [[VSCALE]], 32
        // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}, {{.*}}, i64 [[VSCALE_SIZE]]

        let eq = svcmpeq_s16(
            __pred,
            svget2_s16::<1>(__c_return_value),
            svget2_s16::<1>(__c_return_value),
        );

        let mut __c_return_value = std::mem::MaybeUninit::uninit();
        svcreate3_s16_wrapper(
            __c_return_value.as_mut_ptr(),
            &raw const x0_val,
            &raw const x0_val,
            &raw const x0_val,
        );
        let __c_return_value = __c_return_value.assume_init();
        // CHECK: call void @svcreate3_s16_wrapper(
        // CHECK-NEXT: [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
        // CHECK-NEXT: [[VSCALE_SIZE:%.*]] = mul i64 [[VSCALE]], 48
        // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}, {{.*}}, i64 [[VSCALE_SIZE]]

        let eq = svcmpeq_s16(
            __pred,
            svget3_s16::<2>(__c_return_value),
            svget3_s16::<2>(__c_return_value),
        );

        let mut __c_return_value = std::mem::MaybeUninit::uninit();
        svcreate4_s16_wrapper(
            __c_return_value.as_mut_ptr(),
            &raw const x0_val,
            &raw const x0_val,
            &raw const x0_val,
            &raw const x0_val,
        );
        let __c_return_value = __c_return_value.assume_init();
        // CHECK: call void @svcreate4_s16_wrapper(
        // CHECK-NEXT: [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
        // CHECK-NEXT: [[VSCALE_SIZE:%.*]] = mul i64 [[VSCALE]], 64
        // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}, {{.*}}, i64 [[VSCALE_SIZE]]

        let eq = svcmpeq_s16(
            __pred,
            svget4_s16::<3>(__c_return_value),
            svget4_s16::<3>(__c_return_value),
        );
    }
}
