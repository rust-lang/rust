#![warn(clippy::transmute_float_to_int)]
#![allow(clippy::missing_transmute_annotations)]
#![feature(f128, f128_const)]
#![feature(f16, f16_const)]

fn float_to_int() {
    let _: u32 = unsafe { std::mem::transmute(1f32) };
    //~^ ERROR: transmute from a `f32` to a `u32`
    //~| NOTE: `-D clippy::transmute-float-to-int` implied by `-D warnings`
    let _: i32 = unsafe { std::mem::transmute(1f32) };
    //~^ ERROR: transmute from a `f32` to a `i32`
    let _: u64 = unsafe { std::mem::transmute(1f64) };
    //~^ ERROR: transmute from a `f64` to a `u64`
    let _: i64 = unsafe { std::mem::transmute(1f64) };
    //~^ ERROR: transmute from a `f64` to a `i64`
    let _: u64 = unsafe { std::mem::transmute(1.0) };
    //~^ ERROR: transmute from a `f64` to a `u64`
    let _: u64 = unsafe { std::mem::transmute(-1.0) };
    //~^ ERROR: transmute from a `f64` to a `u64`
}

mod issue_5747 {
    const VALUE16: i16 = unsafe { std::mem::transmute(1f16) };
    //~^ ERROR: transmute from a `f16` to a `i16`
    const VALUE32: i32 = unsafe { std::mem::transmute(1f32) };
    //~^ ERROR: transmute from a `f32` to a `i32`
    const VALUE64: u64 = unsafe { std::mem::transmute(1f64) };
    //~^ ERROR: transmute from a `f64` to a `u64`
    const VALUE128: u128 = unsafe { std::mem::transmute(1f128) };
    //~^ ERROR: transmute from a `f128` to a `u128`

    const fn to_bits_16(v: f16) -> u16 {
        unsafe { std::mem::transmute(v) }
        //~^ ERROR: transmute from a `f16` to a `u16`
    }

    const fn to_bits_32(v: f32) -> u32 {
        unsafe { std::mem::transmute(v) }
        //~^ ERROR: transmute from a `f32` to a `u32`
    }

    const fn to_bits_64(v: f64) -> i64 {
        unsafe { std::mem::transmute(v) }
        //~^ ERROR: transmute from a `f64` to a `i64`
    }

    const fn to_bits_128(v: f128) -> i128 {
        unsafe { std::mem::transmute(v) }
        //~^ ERROR: transmute from a `f128` to a `i128`
    }
}

fn main() {}
