// Test that the simd_bitmask intrinsic produces ok-ish error
// messages when misused.

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x2(pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub u32, pub u32, pub u32, pub u32);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x8(
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x16(
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x32(
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
);

#[repr(simd)]
#[derive(Copy, Clone)]
struct u8x64(
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
);

extern "platform-intrinsic" {
    fn simd_bitmask<T, U>(x: T) -> U;
}

fn main() {
    let m2 = u32x2(0, 0);
    let m4 = u32x4(0, 0, 0, 0);
    let m8 = u8x8(0, 0, 0, 0, 0, 0, 0, 0);
    let m16 = u8x16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let m32 = u8x32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let m64 = u8x64(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    unsafe {
        let _: u8 = simd_bitmask(m2);
        let _: u8 = simd_bitmask(m4);
        let _: u8 = simd_bitmask(m8);
        let _: u16 = simd_bitmask(m16);
        let _: u32 = simd_bitmask(m32);
        let _: u64 = simd_bitmask(m64);

        let _: u16 = simd_bitmask(m2);
        //~^ ERROR bitmask `u16`, expected `u8`

        let _: u16 = simd_bitmask(m8);
        //~^ ERROR bitmask `u16`, expected `u8`

        let _: u32 = simd_bitmask(m16);
        //~^ ERROR bitmask `u32`, expected `u16`

        let _: u64 = simd_bitmask(m32);
        //~^ ERROR bitmask `u64`, expected `u32`

        let _: u128 = simd_bitmask(m64);
        //~^ ERROR bitmask `u128`, expected `u64`

   }
}
