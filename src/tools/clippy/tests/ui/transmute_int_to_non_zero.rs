#![warn(clippy::transmute_int_to_non_zero)]
#![allow(clippy::missing_transmute_annotations)]

use core::num::NonZero;

fn main() {
    let int_u8: u8 = 1;
    let int_u16: u16 = 1;
    let int_u32: u32 = 1;
    let int_u64: u64 = 1;
    let int_u128: u128 = 1;

    let int_i8: i8 = 1;
    let int_i16: i16 = 1;
    let int_i32: i32 = 1;
    let int_i64: i64 = 1;
    let int_i128: i128 = 1;

    let _: NonZero<u8> = unsafe { std::mem::transmute(int_u8) };
    //~^ ERROR: transmute from a `u8` to a `NonZero<u8>`
    //~| NOTE: `-D clippy::transmute-int-to-non-zero` implied by `-D warnings`
    let _: NonZero<u16> = unsafe { std::mem::transmute(int_u16) };
    //~^ ERROR: transmute from a `u16` to a `NonZero<u16>`
    let _: NonZero<u32> = unsafe { std::mem::transmute(int_u32) };
    //~^ ERROR: transmute from a `u32` to a `NonZero<u32>`
    let _: NonZero<u64> = unsafe { std::mem::transmute(int_u64) };
    //~^ ERROR: transmute from a `u64` to a `NonZero<u64>`
    let _: NonZero<u128> = unsafe { std::mem::transmute(int_u128) };
    //~^ ERROR: transmute from a `u128` to a `NonZero<u128>`

    let _: NonZero<i8> = unsafe { std::mem::transmute(int_i8) };
    //~^ ERROR: transmute from a `i8` to a `NonZero<i8>`
    let _: NonZero<i16> = unsafe { std::mem::transmute(int_i16) };
    //~^ ERROR: transmute from a `i16` to a `NonZero<i16>`
    let _: NonZero<i32> = unsafe { std::mem::transmute(int_i32) };
    //~^ ERROR: transmute from a `i32` to a `NonZero<i32>`
    let _: NonZero<i64> = unsafe { std::mem::transmute(int_i64) };
    //~^ ERROR: transmute from a `i64` to a `NonZero<i64>`
    let _: NonZero<i128> = unsafe { std::mem::transmute(int_i128) };
    //~^ ERROR: transmute from a `i128` to a `NonZero<i128>`

    let _: NonZero<u8> = unsafe { NonZero::new_unchecked(int_u8) };
    let _: NonZero<u16> = unsafe { NonZero::new_unchecked(int_u16) };
    let _: NonZero<u32> = unsafe { NonZero::new_unchecked(int_u32) };
    let _: NonZero<u64> = unsafe { NonZero::new_unchecked(int_u64) };
    let _: NonZero<u128> = unsafe { NonZero::new_unchecked(int_u128) };

    let _: NonZero<i8> = unsafe { NonZero::new_unchecked(int_i8) };
    let _: NonZero<i16> = unsafe { NonZero::new_unchecked(int_i16) };
    let _: NonZero<i32> = unsafe { NonZero::new_unchecked(int_i32) };
    let _: NonZero<i64> = unsafe { NonZero::new_unchecked(int_i64) };
    let _: NonZero<i128> = unsafe { NonZero::new_unchecked(int_i128) };
}
