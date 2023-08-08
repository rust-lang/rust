// compile-flags: -C no-prepopulate-passes
//

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]
#![deny(unused)]

// signed integer types

#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x2(i8, i8);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x4(i8, i8, i8, i8);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x8(
    i8, i8, i8, i8, i8, i8, i8, i8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x16(
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x32(
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i8x64(
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct i16x2(i16, i16);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i16x4(i16, i16, i16, i16);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i16x8(
    i16, i16, i16, i16, i16, i16, i16, i16,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i16x16(
    i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i16x32(
    i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct i32x2(i32, i32);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i32x4(i32, i32, i32, i32);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i32x8(
    i32, i32, i32, i32, i32, i32, i32, i32,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i32x16(
    i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct i64x2(i64, i64);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i64x4(i64, i64, i64, i64);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i64x8(
    i64, i64, i64, i64, i64, i64, i64, i64,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct i128x2(i128, i128);
#[repr(simd)] #[derive(Copy, Clone)] pub struct i128x4(i128, i128, i128, i128);

// unsigned integer types

#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x2(u8, u8);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x4(u8, u8, u8, u8);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x8(
    u8, u8, u8, u8, u8, u8, u8, u8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x16(
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x32(
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u8x64(
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct u16x2(u16, u16);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u16x4(u16, u16, u16, u16);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u16x8(
    u16, u16, u16, u16, u16, u16, u16, u16,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u16x16(
    u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u16x32(
    u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct u32x2(u32, u32);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u32x4(u32, u32, u32, u32);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u32x8(
    u32, u32, u32, u32, u32, u32, u32, u32,
);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u32x16(
    u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct u64x2(u64, u64);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u64x4(u64, u64, u64, u64);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u64x8(
    u64, u64, u64, u64, u64, u64, u64, u64,
);

#[repr(simd)] #[derive(Copy, Clone)] pub struct u128x2(u128, u128);
#[repr(simd)] #[derive(Copy, Clone)] pub struct u128x4(u128, u128, u128, u128);

extern "platform-intrinsic" {
    fn simd_saturating_add<T>(x: T, y: T) -> T;
    fn simd_saturating_sub<T>(x: T, y: T) -> T;
}

// NOTE(eddyb) `%{{x|0}}` is used because on some targets (e.g. WASM)
// SIMD vectors are passed directly, resulting in `%x` being a vector,
// while on others they're passed indirectly, resulting in `%x` being
// a pointer to a vector, and `%0` a vector loaded from that pointer.
// This is controlled by the target spec option `simd_types_indirect`.
// The same applies to `%{{y|1}}` as well.

// CHECK-LABEL: @sadd_i8x2
#[no_mangle]
pub unsafe fn sadd_i8x2(x: i8x2, y: i8x2) -> i8x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i8> @llvm.sadd.sat.v2i8(<2 x i8> %{{x|0}}, <2 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i8x4
#[no_mangle]
pub unsafe fn sadd_i8x4(x: i8x4, y: i8x4) -> i8x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i8> @llvm.sadd.sat.v4i8(<4 x i8> %{{x|0}}, <4 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i8x8
#[no_mangle]
pub unsafe fn sadd_i8x8(x: i8x8, y: i8x8) -> i8x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i8> @llvm.sadd.sat.v8i8(<8 x i8> %{{x|0}}, <8 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i8x16
#[no_mangle]
pub unsafe fn sadd_i8x16(x: i8x16, y: i8x16) -> i8x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8> %{{x|0}}, <16 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i8x32
#[no_mangle]
pub unsafe fn sadd_i8x32(x: i8x32, y: i8x32) -> i8x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i8> @llvm.sadd.sat.v32i8(<32 x i8> %{{x|0}}, <32 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i8x64
#[no_mangle]
pub unsafe fn sadd_i8x64(x: i8x64, y: i8x64) -> i8x64 {
    // CHECK: %{{[0-9]+}} = call <64 x i8> @llvm.sadd.sat.v64i8(<64 x i8> %{{x|0}}, <64 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i16x2
#[no_mangle]
pub unsafe fn sadd_i16x2(x: i16x2, y: i16x2) -> i16x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i16> @llvm.sadd.sat.v2i16(<2 x i16> %{{x|0}}, <2 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i16x4
#[no_mangle]
pub unsafe fn sadd_i16x4(x: i16x4, y: i16x4) -> i16x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i16> @llvm.sadd.sat.v4i16(<4 x i16> %{{x|0}}, <4 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i16x8
#[no_mangle]
pub unsafe fn sadd_i16x8(x: i16x8, y: i16x8) -> i16x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %{{x|0}}, <8 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i16x16
#[no_mangle]
pub unsafe fn sadd_i16x16(x: i16x16, y: i16x16) -> i16x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i16> @llvm.sadd.sat.v16i16(<16 x i16> %{{x|0}}, <16 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i16x32
#[no_mangle]
pub unsafe fn sadd_i16x32(x: i16x32, y: i16x32) -> i16x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i16> @llvm.sadd.sat.v32i16(<32 x i16> %{{x|0}}, <32 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i32x2
#[no_mangle]
pub unsafe fn sadd_i32x2(x: i32x2, y: i32x2) -> i32x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i32> @llvm.sadd.sat.v2i32(<2 x i32> %{{x|0}}, <2 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i32x4
#[no_mangle]
pub unsafe fn sadd_i32x4(x: i32x4, y: i32x4) -> i32x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %{{x|0}}, <4 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i32x8
#[no_mangle]
pub unsafe fn sadd_i32x8(x: i32x8, y: i32x8) -> i32x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i32> @llvm.sadd.sat.v8i32(<8 x i32> %{{x|0}}, <8 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i32x16
#[no_mangle]
pub unsafe fn sadd_i32x16(x: i32x16, y: i32x16) -> i32x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i32> @llvm.sadd.sat.v16i32(<16 x i32> %{{x|0}}, <16 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i64x2
#[no_mangle]
pub unsafe fn sadd_i64x2(x: i64x2, y: i64x2) -> i64x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64> %{{x|0}}, <2 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i64x4
#[no_mangle]
pub unsafe fn sadd_i64x4(x: i64x4, y: i64x4) -> i64x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i64> @llvm.sadd.sat.v4i64(<4 x i64> %{{x|0}}, <4 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i64x8
#[no_mangle]
pub unsafe fn sadd_i64x8(x: i64x8, y: i64x8) -> i64x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i64> @llvm.sadd.sat.v8i64(<8 x i64> %{{x|0}}, <8 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i128x2
#[no_mangle]
pub unsafe fn sadd_i128x2(x: i128x2, y: i128x2) -> i128x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i128> @llvm.sadd.sat.v2i128(<2 x i128> %{{x|0}}, <2 x i128> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @sadd_i128x4
#[no_mangle]
pub unsafe fn sadd_i128x4(x: i128x4, y: i128x4) -> i128x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i128> @llvm.sadd.sat.v4i128(<4 x i128> %{{x|0}}, <4 x i128> %{{y|1}})
    simd_saturating_add(x, y)
}



// CHECK-LABEL: @uadd_u8x2
#[no_mangle]
pub unsafe fn uadd_u8x2(x: u8x2, y: u8x2) -> u8x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i8> @llvm.uadd.sat.v2i8(<2 x i8> %{{x|0}}, <2 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u8x4
#[no_mangle]
pub unsafe fn uadd_u8x4(x: u8x4, y: u8x4) -> u8x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i8> @llvm.uadd.sat.v4i8(<4 x i8> %{{x|0}}, <4 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u8x8
#[no_mangle]
pub unsafe fn uadd_u8x8(x: u8x8, y: u8x8) -> u8x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i8> @llvm.uadd.sat.v8i8(<8 x i8> %{{x|0}}, <8 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u8x16
#[no_mangle]
pub unsafe fn uadd_u8x16(x: u8x16, y: u8x16) -> u8x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{x|0}}, <16 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u8x32
#[no_mangle]
pub unsafe fn uadd_u8x32(x: u8x32, y: u8x32) -> u8x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{x|0}}, <32 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u8x64
#[no_mangle]
pub unsafe fn uadd_u8x64(x: u8x64, y: u8x64) -> u8x64 {
    // CHECK: %{{[0-9]+}} = call <64 x i8> @llvm.uadd.sat.v64i8(<64 x i8> %{{x|0}}, <64 x i8> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u16x2
#[no_mangle]
pub unsafe fn uadd_u16x2(x: u16x2, y: u16x2) -> u16x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i16> @llvm.uadd.sat.v2i16(<2 x i16> %{{x|0}}, <2 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u16x4
#[no_mangle]
pub unsafe fn uadd_u16x4(x: u16x4, y: u16x4) -> u16x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i16> @llvm.uadd.sat.v4i16(<4 x i16> %{{x|0}}, <4 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u16x8
#[no_mangle]
pub unsafe fn uadd_u16x8(x: u16x8, y: u16x8) -> u16x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{x|0}}, <8 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u16x16
#[no_mangle]
pub unsafe fn uadd_u16x16(x: u16x16, y: u16x16) -> u16x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{x|0}}, <16 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u16x32
#[no_mangle]
pub unsafe fn uadd_u16x32(x: u16x32, y: u16x32) -> u16x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i16> @llvm.uadd.sat.v32i16(<32 x i16> %{{x|0}}, <32 x i16> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u32x2
#[no_mangle]
pub unsafe fn uadd_u32x2(x: u32x2, y: u32x2) -> u32x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i32> @llvm.uadd.sat.v2i32(<2 x i32> %{{x|0}}, <2 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u32x4
#[no_mangle]
pub unsafe fn uadd_u32x4(x: u32x4, y: u32x4) -> u32x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %{{x|0}}, <4 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u32x8
#[no_mangle]
pub unsafe fn uadd_u32x8(x: u32x8, y: u32x8) -> u32x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i32> @llvm.uadd.sat.v8i32(<8 x i32> %{{x|0}}, <8 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u32x16
#[no_mangle]
pub unsafe fn uadd_u32x16(x: u32x16, y: u32x16) -> u32x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i32> @llvm.uadd.sat.v16i32(<16 x i32> %{{x|0}}, <16 x i32> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u64x2
#[no_mangle]
pub unsafe fn uadd_u64x2(x: u64x2, y: u64x2) -> u64x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.uadd.sat.v2i64(<2 x i64> %{{x|0}}, <2 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u64x4
#[no_mangle]
pub unsafe fn uadd_u64x4(x: u64x4, y: u64x4) -> u64x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i64> @llvm.uadd.sat.v4i64(<4 x i64> %{{x|0}}, <4 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u64x8
#[no_mangle]
pub unsafe fn uadd_u64x8(x: u64x8, y: u64x8) -> u64x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i64> @llvm.uadd.sat.v8i64(<8 x i64> %{{x|0}}, <8 x i64> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u128x2
#[no_mangle]
pub unsafe fn uadd_u128x2(x: u128x2, y: u128x2) -> u128x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i128> @llvm.uadd.sat.v2i128(<2 x i128> %{{x|0}}, <2 x i128> %{{y|1}})
    simd_saturating_add(x, y)
}

// CHECK-LABEL: @uadd_u128x4
#[no_mangle]
pub unsafe fn uadd_u128x4(x: u128x4, y: u128x4) -> u128x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i128> @llvm.uadd.sat.v4i128(<4 x i128> %{{x|0}}, <4 x i128> %{{y|1}})
    simd_saturating_add(x, y)
}





// CHECK-LABEL: @ssub_i8x2
#[no_mangle]
pub unsafe fn ssub_i8x2(x: i8x2, y: i8x2) -> i8x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i8> @llvm.ssub.sat.v2i8(<2 x i8> %{{x|0}}, <2 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i8x4
#[no_mangle]
pub unsafe fn ssub_i8x4(x: i8x4, y: i8x4) -> i8x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i8> @llvm.ssub.sat.v4i8(<4 x i8> %{{x|0}}, <4 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i8x8
#[no_mangle]
pub unsafe fn ssub_i8x8(x: i8x8, y: i8x8) -> i8x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i8> @llvm.ssub.sat.v8i8(<8 x i8> %{{x|0}}, <8 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i8x16
#[no_mangle]
pub unsafe fn ssub_i8x16(x: i8x16, y: i8x16) -> i8x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.ssub.sat.v16i8(<16 x i8> %{{x|0}}, <16 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i8x32
#[no_mangle]
pub unsafe fn ssub_i8x32(x: i8x32, y: i8x32) -> i8x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i8> @llvm.ssub.sat.v32i8(<32 x i8> %{{x|0}}, <32 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i8x64
#[no_mangle]
pub unsafe fn ssub_i8x64(x: i8x64, y: i8x64) -> i8x64 {
    // CHECK: %{{[0-9]+}} = call <64 x i8> @llvm.ssub.sat.v64i8(<64 x i8> %{{x|0}}, <64 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i16x2
#[no_mangle]
pub unsafe fn ssub_i16x2(x: i16x2, y: i16x2) -> i16x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i16> @llvm.ssub.sat.v2i16(<2 x i16> %{{x|0}}, <2 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i16x4
#[no_mangle]
pub unsafe fn ssub_i16x4(x: i16x4, y: i16x4) -> i16x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> %{{x|0}}, <4 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i16x8
#[no_mangle]
pub unsafe fn ssub_i16x8(x: i16x8, y: i16x8) -> i16x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %{{x|0}}, <8 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i16x16
#[no_mangle]
pub unsafe fn ssub_i16x16(x: i16x16, y: i16x16) -> i16x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i16> @llvm.ssub.sat.v16i16(<16 x i16> %{{x|0}}, <16 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i16x32
#[no_mangle]
pub unsafe fn ssub_i16x32(x: i16x32, y: i16x32) -> i16x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i16> @llvm.ssub.sat.v32i16(<32 x i16> %{{x|0}}, <32 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i32x2
#[no_mangle]
pub unsafe fn ssub_i32x2(x: i32x2, y: i32x2) -> i32x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> %{{x|0}}, <2 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i32x4
#[no_mangle]
pub unsafe fn ssub_i32x4(x: i32x4, y: i32x4) -> i32x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %{{x|0}}, <4 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i32x8
#[no_mangle]
pub unsafe fn ssub_i32x8(x: i32x8, y: i32x8) -> i32x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i32> @llvm.ssub.sat.v8i32(<8 x i32> %{{x|0}}, <8 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i32x16
#[no_mangle]
pub unsafe fn ssub_i32x16(x: i32x16, y: i32x16) -> i32x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i32> @llvm.ssub.sat.v16i32(<16 x i32> %{{x|0}}, <16 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i64x2
#[no_mangle]
pub unsafe fn ssub_i64x2(x: i64x2, y: i64x2) -> i64x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64> %{{x|0}}, <2 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i64x4
#[no_mangle]
pub unsafe fn ssub_i64x4(x: i64x4, y: i64x4) -> i64x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i64> @llvm.ssub.sat.v4i64(<4 x i64> %{{x|0}}, <4 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i64x8
#[no_mangle]
pub unsafe fn ssub_i64x8(x: i64x8, y: i64x8) -> i64x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i64> @llvm.ssub.sat.v8i64(<8 x i64> %{{x|0}}, <8 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i128x2
#[no_mangle]
pub unsafe fn ssub_i128x2(x: i128x2, y: i128x2) -> i128x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i128> @llvm.ssub.sat.v2i128(<2 x i128> %{{x|0}}, <2 x i128> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @ssub_i128x4
#[no_mangle]
pub unsafe fn ssub_i128x4(x: i128x4, y: i128x4) -> i128x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i128> @llvm.ssub.sat.v4i128(<4 x i128> %{{x|0}}, <4 x i128> %{{y|1}})
    simd_saturating_sub(x, y)
}



// CHECK-LABEL: @usub_u8x2
#[no_mangle]
pub unsafe fn usub_u8x2(x: u8x2, y: u8x2) -> u8x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i8> @llvm.usub.sat.v2i8(<2 x i8> %{{x|0}}, <2 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u8x4
#[no_mangle]
pub unsafe fn usub_u8x4(x: u8x4, y: u8x4) -> u8x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i8> @llvm.usub.sat.v4i8(<4 x i8> %{{x|0}}, <4 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u8x8
#[no_mangle]
pub unsafe fn usub_u8x8(x: u8x8, y: u8x8) -> u8x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i8> @llvm.usub.sat.v8i8(<8 x i8> %{{x|0}}, <8 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u8x16
#[no_mangle]
pub unsafe fn usub_u8x16(x: u8x16, y: u8x16) -> u8x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{x|0}}, <16 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u8x32
#[no_mangle]
pub unsafe fn usub_u8x32(x: u8x32, y: u8x32) -> u8x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{x|0}}, <32 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u8x64
#[no_mangle]
pub unsafe fn usub_u8x64(x: u8x64, y: u8x64) -> u8x64 {
    // CHECK: %{{[0-9]+}} = call <64 x i8> @llvm.usub.sat.v64i8(<64 x i8> %{{x|0}}, <64 x i8> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u16x2
#[no_mangle]
pub unsafe fn usub_u16x2(x: u16x2, y: u16x2) -> u16x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i16> @llvm.usub.sat.v2i16(<2 x i16> %{{x|0}}, <2 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u16x4
#[no_mangle]
pub unsafe fn usub_u16x4(x: u16x4, y: u16x4) -> u16x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i16> @llvm.usub.sat.v4i16(<4 x i16> %{{x|0}}, <4 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u16x8
#[no_mangle]
pub unsafe fn usub_u16x8(x: u16x8, y: u16x8) -> u16x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{x|0}}, <8 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u16x16
#[no_mangle]
pub unsafe fn usub_u16x16(x: u16x16, y: u16x16) -> u16x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{x|0}}, <16 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u16x32
#[no_mangle]
pub unsafe fn usub_u16x32(x: u16x32, y: u16x32) -> u16x32 {
    // CHECK: %{{[0-9]+}} = call <32 x i16> @llvm.usub.sat.v32i16(<32 x i16> %{{x|0}}, <32 x i16> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u32x2
#[no_mangle]
pub unsafe fn usub_u32x2(x: u32x2, y: u32x2) -> u32x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i32> @llvm.usub.sat.v2i32(<2 x i32> %{{x|0}}, <2 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u32x4
#[no_mangle]
pub unsafe fn usub_u32x4(x: u32x4, y: u32x4) -> u32x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %{{x|0}}, <4 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u32x8
#[no_mangle]
pub unsafe fn usub_u32x8(x: u32x8, y: u32x8) -> u32x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i32> @llvm.usub.sat.v8i32(<8 x i32> %{{x|0}}, <8 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u32x16
#[no_mangle]
pub unsafe fn usub_u32x16(x: u32x16, y: u32x16) -> u32x16 {
    // CHECK: %{{[0-9]+}} = call <16 x i32> @llvm.usub.sat.v16i32(<16 x i32> %{{x|0}}, <16 x i32> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u64x2
#[no_mangle]
pub unsafe fn usub_u64x2(x: u64x2, y: u64x2) -> u64x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i64> @llvm.usub.sat.v2i64(<2 x i64> %{{x|0}}, <2 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u64x4
#[no_mangle]
pub unsafe fn usub_u64x4(x: u64x4, y: u64x4) -> u64x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i64> @llvm.usub.sat.v4i64(<4 x i64> %{{x|0}}, <4 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u64x8
#[no_mangle]
pub unsafe fn usub_u64x8(x: u64x8, y: u64x8) -> u64x8 {
    // CHECK: %{{[0-9]+}} = call <8 x i64> @llvm.usub.sat.v8i64(<8 x i64> %{{x|0}}, <8 x i64> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u128x2
#[no_mangle]
pub unsafe fn usub_u128x2(x: u128x2, y: u128x2) -> u128x2 {
    // CHECK: %{{[0-9]+}} = call <2 x i128> @llvm.usub.sat.v2i128(<2 x i128> %{{x|0}}, <2 x i128> %{{y|1}})
    simd_saturating_sub(x, y)
}

// CHECK-LABEL: @usub_u128x4
#[no_mangle]
pub unsafe fn usub_u128x4(x: u128x4, y: u128x4) -> u128x4 {
    // CHECK: %{{[0-9]+}} = call <4 x i128> @llvm.usub.sat.v4i128(<4 x i128> %{{x|0}}, <4 x i128> %{{y|1}})
    simd_saturating_sub(x, y)
}
