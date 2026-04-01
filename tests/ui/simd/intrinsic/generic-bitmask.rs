//@ build-fail

// Test that the simd_bitmask intrinsic produces ok-ish error
// messages when misused.

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::simd_bitmask;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x2([u32; 2]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4([u32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x8([u8; 8]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x16([u8; 16]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x32([u8; 32]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x64([u8; 64]);

fn main() {
    let m2 = u32x2([0; 2]);
    let m4 = u32x4([0; 4]);
    let m8 = u8x8([0; 8]);
    let m16 = u8x16([0; 16]);
    let m32 = u8x32([0; 32]);
    let m64 = u8x64([0; 64]);

    unsafe {
        let _: u8 = simd_bitmask(m2);
        let _: u8 = simd_bitmask(m4);
        let _: u8 = simd_bitmask(m8);
        let _: u16 = simd_bitmask(m16);
        let _: u32 = simd_bitmask(m32);
        let _: u64 = simd_bitmask(m64);

        let _: u16 = simd_bitmask(m2);
        //~^ ERROR invalid monomorphization of `simd_bitmask` intrinsic

        let _: u16 = simd_bitmask(m8);
        //~^ ERROR invalid monomorphization of `simd_bitmask` intrinsic

        let _: u32 = simd_bitmask(m16);
        //~^ ERROR invalid monomorphization of `simd_bitmask` intrinsic

        let _: u64 = simd_bitmask(m32);
        //~^ ERROR invalid monomorphization of `simd_bitmask` intrinsic

        let _: u128 = simd_bitmask(m64);
        //~^ ERROR invalid monomorphization of `simd_bitmask` intrinsic
    }
}
