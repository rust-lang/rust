//@ build-fail
//@ ignore-backends: gcc

#![feature(repr_simd, core_intrinsics)]
#![allow(non_camel_case_types)]

use std::intrinsics::simd::*;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i32x4(pub [i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct u32x4(pub [u32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct f32x4(pub [f32; 4]);

fn main() {
    let x = i32x4([0, 0, 0, 0]);
    let y = u32x4([0, 0, 0, 0]);
    let z = f32x4([0.0, 0.0, 0.0, 0.0]);

    unsafe {
        simd_add(x, x);
        simd_add(y, y);
        simd_add(z, z);
        simd_sub(x, x);
        simd_sub(y, y);
        simd_sub(z, z);
        simd_mul(x, x);
        simd_mul(y, y);
        simd_mul(z, z);
        simd_div(x, x);
        simd_div(y, y);
        simd_div(z, z);
        simd_rem(x, x);
        simd_rem(y, y);
        simd_rem(z, z);

        simd_shl(x, x);
        simd_shl(y, y);
        simd_shr(x, x);
        simd_shr(y, y);
        simd_funnel_shl(x, x, x);
        simd_funnel_shl(y, y, y);
        simd_funnel_shr(x, x, x);
        simd_funnel_shr(y, y, y);
        simd_and(x, x);
        simd_and(y, y);
        simd_or(x, x);
        simd_or(y, y);
        simd_xor(x, x);
        simd_xor(y, y);

        simd_neg(x);
        simd_neg(z);
        simd_bswap(x);
        simd_bswap(y);
        simd_bitreverse(x);
        simd_bitreverse(y);
        simd_ctlz(x);
        simd_ctlz(y);
        simd_cttz(x);
        simd_cttz(y);

        simd_add(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_sub(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_mul(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_div(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_shl(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_shr(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_funnel_shl(0, 0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_funnel_shr(0, 0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_and(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_or(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_xor(0, 0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_neg(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_bswap(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_bitreverse(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_ctlz(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_cttz(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`

        simd_shl(z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_shr(z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_funnel_shl(z, z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_funnel_shr(z, z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_and(z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_or(z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_xor(z, z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_bswap(z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_bitreverse(z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_ctlz(z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_ctpop(z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
        simd_cttz(z);
        //~^ ERROR unsupported operation on `f32x4` with element `f32`
    }
}
