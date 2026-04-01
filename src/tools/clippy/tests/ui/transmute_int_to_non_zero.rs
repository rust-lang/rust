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
    //~^ transmute_int_to_non_zero

    let _: NonZero<u16> = unsafe { std::mem::transmute(int_u16) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<u32> = unsafe { std::mem::transmute(int_u32) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<u64> = unsafe { std::mem::transmute(int_u64) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<u128> = unsafe { std::mem::transmute(int_u128) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<i8> = unsafe { std::mem::transmute(int_i8) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<i16> = unsafe { std::mem::transmute(int_i16) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<i32> = unsafe { std::mem::transmute(int_i32) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<i64> = unsafe { std::mem::transmute(int_i64) };
    //~^ transmute_int_to_non_zero

    let _: NonZero<i128> = unsafe { std::mem::transmute(int_i128) };
    //~^ transmute_int_to_non_zero

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
