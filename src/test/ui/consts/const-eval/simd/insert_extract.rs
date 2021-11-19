// run-pass
#![feature(repr_simd)]
#![feature(platform_intrinsics)]
#![feature(staged_api)]
#![stable(feature = "foo", since = "1.33.7")]
#![allow(non_camel_case_types)]

#[repr(simd)] struct i8x1(i8);
#[repr(simd)] struct u16x2(u16, u16);
// Make some of them array types to ensure those also work.
#[repr(simd)] struct i8x1_arr([i8; 1]);
#[repr(simd)] struct f32x4([f32; 4]);

extern "platform-intrinsic" {
    #[rustc_const_stable(feature = "foo", since = "1.3.37")]
    fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;
    #[rustc_const_stable(feature = "foo", since = "1.3.37")]
    fn simd_extract<T, U>(x: T, idx: u32) -> U;
}

fn main() {
    {
        const U: i8x1 = i8x1(13);
        const V: i8x1 = unsafe { simd_insert(U, 0_u32, 42_i8) };
        const X0: i8 = V.0;
        const Y0: i8 = unsafe { simd_extract(V, 0) };
        assert_eq!(X0, 42);
        assert_eq!(Y0, 42);
    }
    {
        const U: i8x1_arr = i8x1_arr([13]);
        const V: i8x1_arr = unsafe { simd_insert(U, 0_u32, 42_i8) };
        const X0: i8 = V.0[0];
        const Y0: i8 = unsafe { simd_extract(V, 0) };
        assert_eq!(X0, 42);
        assert_eq!(Y0, 42);
    }
    {
        const U: u16x2 = u16x2(13, 14);
        const V: u16x2 = unsafe { simd_insert(U, 1_u32, 42_u16) };
        const X0: u16 = V.0;
        const X1: u16 = V.1;
        const Y0: u16 = unsafe { simd_extract(V, 0) };
        const Y1: u16 = unsafe { simd_extract(V, 1) };
        assert_eq!(X0, 13);
        assert_eq!(X1, 42);
        assert_eq!(Y0, 13);
        assert_eq!(Y1, 42);
    }
    {
        const U: f32x4 = f32x4([13., 14., 15., 16.]);
        const V: f32x4 = unsafe { simd_insert(U, 1_u32, 42_f32) };
        const X0: f32 = V.0[0];
        const X1: f32 = V.0[1];
        const X2: f32 = V.0[2];
        const X3: f32 = V.0[3];
        const Y0: f32 = unsafe { simd_extract(V, 0) };
        const Y1: f32 = unsafe { simd_extract(V, 1) };
        const Y2: f32 = unsafe { simd_extract(V, 2) };
        const Y3: f32 = unsafe { simd_extract(V, 3) };
        assert_eq!(X0, 13.);
        assert_eq!(X1, 42.);
        assert_eq!(X2, 15.);
        assert_eq!(X3, 16.);
        assert_eq!(Y0, 13.);
        assert_eq!(Y1, 42.);
        assert_eq!(Y2, 15.);
        assert_eq!(Y3, 16.);
    }
}
