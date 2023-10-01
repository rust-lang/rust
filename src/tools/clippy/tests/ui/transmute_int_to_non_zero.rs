#![warn(clippy::transmute_int_to_non_zero)]

use core::num::*;

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

    let _: NonZeroU8 = unsafe { std::mem::transmute(int_u8) };
    //~^ ERROR: transmute from a `u8` to a `NonZeroU8`
    //~| NOTE: `-D clippy::transmute-int-to-non-zero` implied by `-D warnings`
    let _: NonZeroU16 = unsafe { std::mem::transmute(int_u16) };
    //~^ ERROR: transmute from a `u16` to a `NonZeroU16`
    let _: NonZeroU32 = unsafe { std::mem::transmute(int_u32) };
    //~^ ERROR: transmute from a `u32` to a `NonZeroU32`
    let _: NonZeroU64 = unsafe { std::mem::transmute(int_u64) };
    //~^ ERROR: transmute from a `u64` to a `NonZeroU64`
    let _: NonZeroU128 = unsafe { std::mem::transmute(int_u128) };
    //~^ ERROR: transmute from a `u128` to a `NonZeroU128`

    let _: NonZeroI8 = unsafe { std::mem::transmute(int_i8) };
    //~^ ERROR: transmute from a `i8` to a `NonZeroI8`
    let _: NonZeroI16 = unsafe { std::mem::transmute(int_i16) };
    //~^ ERROR: transmute from a `i16` to a `NonZeroI16`
    let _: NonZeroI32 = unsafe { std::mem::transmute(int_i32) };
    //~^ ERROR: transmute from a `i32` to a `NonZeroI32`
    let _: NonZeroI64 = unsafe { std::mem::transmute(int_i64) };
    //~^ ERROR: transmute from a `i64` to a `NonZeroI64`
    let _: NonZeroI128 = unsafe { std::mem::transmute(int_i128) };
    //~^ ERROR: transmute from a `i128` to a `NonZeroI128`

    let _: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(int_u8) };
    let _: NonZeroU16 = unsafe { NonZeroU16::new_unchecked(int_u16) };
    let _: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(int_u32) };
    let _: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(int_u64) };
    let _: NonZeroU128 = unsafe { NonZeroU128::new_unchecked(int_u128) };

    let _: NonZeroI8 = unsafe { NonZeroI8::new_unchecked(int_i8) };
    let _: NonZeroI16 = unsafe { NonZeroI16::new_unchecked(int_i16) };
    let _: NonZeroI32 = unsafe { NonZeroI32::new_unchecked(int_i32) };
    let _: NonZeroI64 = unsafe { NonZeroI64::new_unchecked(int_i64) };
    let _: NonZeroI128 = unsafe { NonZeroI128::new_unchecked(int_i128) };
}
