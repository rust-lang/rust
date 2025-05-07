#![warn(clippy::transmute_float_to_int)]
#![allow(clippy::missing_transmute_annotations, unnecessary_transmutes)]
#![feature(f128)]
#![feature(f16)]

fn float_to_int() {
    let _: u32 = unsafe { std::mem::transmute(1f32) };
    //~^ transmute_float_to_int

    let _: i32 = unsafe { std::mem::transmute(1f32) };
    //~^ transmute_float_to_int

    let _: u64 = unsafe { std::mem::transmute(1f64) };
    //~^ transmute_float_to_int

    let _: i64 = unsafe { std::mem::transmute(1f64) };
    //~^ transmute_float_to_int

    let _: u64 = unsafe { std::mem::transmute(1.0) };
    //~^ transmute_float_to_int

    let _: u64 = unsafe { std::mem::transmute(-1.0) };
    //~^ transmute_float_to_int
}

mod issue_5747 {
    const VALUE16: i16 = unsafe { std::mem::transmute(1f16) };
    //~^ transmute_float_to_int

    const VALUE32: i32 = unsafe { std::mem::transmute(1f32) };
    //~^ transmute_float_to_int

    const VALUE64: u64 = unsafe { std::mem::transmute(1f64) };
    //~^ transmute_float_to_int

    const VALUE128: u128 = unsafe { std::mem::transmute(1f128) };
    //~^ transmute_float_to_int

    const fn to_bits_16(v: f16) -> u16 {
        unsafe { std::mem::transmute(v) }
        //~^ transmute_float_to_int
    }

    const fn to_bits_32(v: f32) -> u32 {
        unsafe { std::mem::transmute(v) }
        //~^ transmute_float_to_int
    }

    const fn to_bits_64(v: f64) -> i64 {
        unsafe { std::mem::transmute(v) }
        //~^ transmute_float_to_int
    }

    const fn to_bits_128(v: f128) -> i128 {
        unsafe { std::mem::transmute(v) }
        //~^ transmute_float_to_int
    }
}

fn main() {}
