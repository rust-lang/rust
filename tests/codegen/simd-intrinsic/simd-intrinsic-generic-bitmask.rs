// compile-flags: -C no-prepopulate-passes
//

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x2(u32, u32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i32x2(i32, i32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i8x16(
    i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8,
);


extern "platform-intrinsic" {
    fn simd_bitmask<T, U>(x: T) -> U;
}

// NOTE(eddyb) `%{{x|1}}` is used because on some targets (e.g. WASM)
// SIMD vectors are passed directly, resulting in `%x` being a vector,
// while on others they're passed indirectly, resulting in `%x` being
// a pointer to a vector, and `%1` a vector loaded from that pointer.
// This is controlled by the target spec option `simd_types_indirect`.

// CHECK-LABEL: @bitmask_int
#[no_mangle]
pub unsafe fn bitmask_int(x: i32x2) -> u8 {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> %{{x|1}}, <i32 31, i32 31>
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: [[C:%[0-9]+]] = bitcast <2 x i1> [[B]] to i2
    // CHECK: %{{[0-9]+}} = zext i2 [[C]] to i8
    simd_bitmask(x)
}

// CHECK-LABEL: @bitmask_uint
#[no_mangle]
pub unsafe fn bitmask_uint(x: u32x2) -> u8 {
    // CHECK: [[A:%[0-9]+]] = lshr <2 x i32> %{{x|1}}, <i32 31, i32 31>
    // CHECK: [[B:%[0-9]+]] = trunc <2 x i32> [[A]] to <2 x i1>
    // CHECK: [[C:%[0-9]+]] = bitcast <2 x i1> [[B]] to i2
    // CHECK: %{{[0-9]+}} = zext i2 [[C]] to i8
    simd_bitmask(x)
}

// CHECK-LABEL: @bitmask_int16
#[no_mangle]
pub unsafe fn bitmask_int16(x: i8x16) -> u16 {
    // CHECK: [[A:%[0-9]+]] = lshr <16 x i8> %{{x|1|2}}, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
    // CHECK: [[B:%[0-9]+]] = trunc <16 x i8> [[A]] to <16 x i1>
    // CHECK: %{{[0-9]+}} = bitcast <16 x i1> [[B]] to i16
    // CHECK-NOT: zext
    simd_bitmask(x)
}
