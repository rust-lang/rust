//@ build-fail

// Test that the simd_select intrinsic produces ok-ish error
// messages when misused.

#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub [f32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub [u32; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq)]
struct b8x4(pub [i8; 4]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq)]
struct b8x8(pub [i8; 8]);

extern "rust-intrinsic" {
    fn simd_select<T, U>(x: T, a: U, b: U) -> U;
    fn simd_select_bitmask<T, U>(x: T, a: U, b: U) -> U;
}

fn main() {
    let m4 = b8x4([0, 0, 0, 0]);
    let m8 = b8x8([0, 0, 0, 0, 0, 0, 0, 0]);
    let x = u32x4([0, 0, 0, 0]);
    let z = f32x4([0.0, 0.0, 0.0, 0.0]);

    unsafe {
        simd_select(m4, x, x);

        simd_select(m8, x, x);
        //~^ ERROR mismatched lengths: mask length `8` != other vector length `4`

        simd_select(x, x, x);
        //~^ ERROR mask element type is `u32`, expected `i_`

        simd_select(z, z, z);
        //~^ ERROR mask element type is `f32`, expected `i_`

        simd_select(m4, 0u32, 1u32);
        //~^ ERROR found non-SIMD `u32`

        simd_select_bitmask(0u16, x, x);
        //~^ ERROR invalid bitmask `u16`, expected `u8` or `[u8; 1]`

        simd_select_bitmask(0u8, 1u32, 2u32);
        //~^ ERROR found non-SIMD `u32`

        simd_select_bitmask(0.0f32, x, x);
        //~^ ERROR invalid bitmask `f32`, expected `u8` or `[u8; 1]`

        simd_select_bitmask("x", x, x);
        //~^ ERROR invalid bitmask `&str`, expected `u8` or `[u8; 1]`
    }
}
