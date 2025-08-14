//@ run-rustfix

// The `try_into` suggestion doesn't include this, but we do suggest it after applying it
use std::convert::TryInto;

#[allow(unused_must_use)]
fn main() {
    let x_usize: usize = 1;
    let x_u128: u128 = 2;
    let x_u64: u64 = 3;
    let x_u32: u32 = 4;
    let x_u16: u16 = 5;
    let x_u8: u8 = 6;
    let x_isize: isize = 7;
    let x_i64: i64 = 8;
    let x_i32: i32 = 9;
    let x_i16: i16 = 10;
    let x_i8: i8 = 11;
    let x_i128: i128 = 12;

    /* u<->u */
    {
        x_u8 > x_u16;
        //~^ ERROR mismatched types
        x_u8 > x_u32;
        //~^ ERROR mismatched types
        x_u8 > x_u64;
        //~^ ERROR mismatched types
        x_u8 > x_u128;
        //~^ ERROR mismatched types
        x_u8 > x_usize;
        //~^ ERROR mismatched types

        x_u16 > x_u8;
        //~^ ERROR mismatched types
        x_u16 > x_u32;
        //~^ ERROR mismatched types
        x_u16 > x_u64;
        //~^ ERROR mismatched types
        x_u16 > x_u128;
        //~^ ERROR mismatched types
        x_u16 > x_usize;
        //~^ ERROR mismatched types

        x_u32 > x_u8;
        //~^ ERROR mismatched types
        x_u32 > x_u16;
        //~^ ERROR mismatched types
        x_u32 > x_u64;
        //~^ ERROR mismatched types
        x_u32 > x_u128;
        //~^ ERROR mismatched types
        x_u32 > x_usize;
        //~^ ERROR mismatched types

        x_u64 > x_u8;
        //~^ ERROR mismatched types
        x_u64 > x_u16;
        //~^ ERROR mismatched types
        x_u64 > x_u32;
        //~^ ERROR mismatched types
        x_u64 > x_u128;
        //~^ ERROR mismatched types
        x_u64 > x_usize;
        //~^ ERROR mismatched types

        x_u128 > x_u8;
        //~^ ERROR mismatched types
        x_u128 > x_u16;
        //~^ ERROR mismatched types
        x_u128 > x_u32;
        //~^ ERROR mismatched types
        x_u128 > x_u64;
        //~^ ERROR mismatched types
        x_u128 > x_usize;
        //~^ ERROR mismatched types

        x_usize > x_u8;
        //~^ ERROR mismatched types
        x_usize > x_u16;
        //~^ ERROR mismatched types
        x_usize > x_u32;
        //~^ ERROR mismatched types
        x_usize > x_u64;
        //~^ ERROR mismatched types
        x_usize > x_u128;
        //~^ ERROR mismatched types
    }

    /* i<->i */
    {
        x_i8 > x_i16;
        //~^ ERROR mismatched types
        x_i8 > x_i32;
        //~^ ERROR mismatched types
        x_i8 > x_i64;
        //~^ ERROR mismatched types
        x_i8 > x_i128;
        //~^ ERROR mismatched types
        x_i8 > x_isize;
        //~^ ERROR mismatched types

        x_i16 > x_i8;
        //~^ ERROR mismatched types
        x_i16 > x_i32;
        //~^ ERROR mismatched types
        x_i16 > x_i64;
        //~^ ERROR mismatched types
        x_i16 > x_i128;
        //~^ ERROR mismatched types
        x_i16 > x_isize;
        //~^ ERROR mismatched types

        x_i32 > x_i8;
        //~^ ERROR mismatched types
        x_i32 > x_i16;
        //~^ ERROR mismatched types
        x_i32 > x_i64;
        //~^ ERROR mismatched types
        x_i32 > x_i128;
        //~^ ERROR mismatched types
        x_i32 > x_isize;
        //~^ ERROR mismatched types

        x_i64 > x_i8;
        //~^ ERROR mismatched types
        x_i64 > x_i16;
        //~^ ERROR mismatched types
        x_i64 > x_i32;
        //~^ ERROR mismatched types
        x_i64 > x_i128;
        //~^ ERROR mismatched types
        x_i64 > x_isize;
        //~^ ERROR mismatched types

        x_i128 > x_i8;
        //~^ ERROR mismatched types
        x_i128 > x_i16;
        //~^ ERROR mismatched types
        x_i128 > x_i32;
        //~^ ERROR mismatched types
        x_i128 > x_i64;
        //~^ ERROR mismatched types
        x_i128 > x_isize;
        //~^ ERROR mismatched types

        x_isize > x_i8;
        //~^ ERROR mismatched types
        x_isize > x_i16;
        //~^ ERROR mismatched types
        x_isize > x_i32;
        //~^ ERROR mismatched types
        x_isize > x_i64;
        //~^ ERROR mismatched types
        x_isize > x_i128;
        //~^ ERROR mismatched types
    }

    /* u<->i */
    {
        x_u8 > x_i8;
        //~^ ERROR mismatched types
        x_u8 > x_i16;
        //~^ ERROR mismatched types
        x_u8 > x_i32;
        //~^ ERROR mismatched types
        x_u8 > x_i64;
        //~^ ERROR mismatched types
        x_u8 > x_i128;
        //~^ ERROR mismatched types
        x_u8 > x_isize;
        //~^ ERROR mismatched types

        x_u16 > x_i8;
        //~^ ERROR mismatched types
        x_u16 > x_i16;
        //~^ ERROR mismatched types
        x_u16 > x_i32;
        //~^ ERROR mismatched types
        x_u16 > x_i64;
        //~^ ERROR mismatched types
        x_u16 > x_i128;
        //~^ ERROR mismatched types
        x_u16 > x_isize;
        //~^ ERROR mismatched types

        x_u32 > x_i8;
        //~^ ERROR mismatched types
        x_u32 > x_i16;
        //~^ ERROR mismatched types
        x_u32 > x_i32;
        //~^ ERROR mismatched types
        x_u32 > x_i64;
        //~^ ERROR mismatched types
        x_u32 > x_i128;
        //~^ ERROR mismatched types
        x_u32 > x_isize;
        //~^ ERROR mismatched types

        x_u64 > x_i8;
        //~^ ERROR mismatched types
        x_u64 > x_i16;
        //~^ ERROR mismatched types
        x_u64 > x_i32;
        //~^ ERROR mismatched types
        x_u64 > x_i64;
        //~^ ERROR mismatched types
        x_u64 > x_i128;
        //~^ ERROR mismatched types
        x_u64 > x_isize;
        //~^ ERROR mismatched types

        x_u128 > x_i8;
        //~^ ERROR mismatched types
        x_u128 > x_i16;
        //~^ ERROR mismatched types
        x_u128 > x_i32;
        //~^ ERROR mismatched types
        x_u128 > x_i64;
        //~^ ERROR mismatched types
        x_u128 > x_i128;
        //~^ ERROR mismatched types
        x_u128 > x_isize;
        //~^ ERROR mismatched types

        x_usize > x_i8;
        //~^ ERROR mismatched types
        x_usize > x_i16;
        //~^ ERROR mismatched types
        x_usize > x_i32;
        //~^ ERROR mismatched types
        x_usize > x_i64;
        //~^ ERROR mismatched types
        x_usize > x_i128;
        //~^ ERROR mismatched types
        x_usize > x_isize;
        //~^ ERROR mismatched types
    }

    /* i<->u */
    {
        x_i8 > x_u8;
        //~^ ERROR mismatched types
        x_i8 > x_u16;
        //~^ ERROR mismatched types
        x_i8 > x_u32;
        //~^ ERROR mismatched types
        x_i8 > x_u64;
        //~^ ERROR mismatched types
        x_i8 > x_u128;
        //~^ ERROR mismatched types
        x_i8 > x_usize;
        //~^ ERROR mismatched types

        x_i16 > x_u8;
        //~^ ERROR mismatched types
        x_i16 > x_u16;
        //~^ ERROR mismatched types
        x_i16 > x_u32;
        //~^ ERROR mismatched types
        x_i16 > x_u64;
        //~^ ERROR mismatched types
        x_i16 > x_u128;
        //~^ ERROR mismatched types
        x_i16 > x_usize;
        //~^ ERROR mismatched types

        x_i32 > x_u8;
        //~^ ERROR mismatched types
        x_i32 > x_u16;
        //~^ ERROR mismatched types
        x_i32 > x_u32;
        //~^ ERROR mismatched types
        x_i32 > x_u64;
        //~^ ERROR mismatched types
        x_i32 > x_u128;
        //~^ ERROR mismatched types
        x_i32 > x_usize;
        //~^ ERROR mismatched types

        x_i64 > x_u8;
        //~^ ERROR mismatched types
        x_i64 > x_u16;
        //~^ ERROR mismatched types
        x_i64 > x_u32;
        //~^ ERROR mismatched types
        x_i64 > x_u64;
        //~^ ERROR mismatched types
        x_i64 > x_u128;
        //~^ ERROR mismatched types
        x_i64 > x_usize;
        //~^ ERROR mismatched types

        x_i128 > x_u8;
        //~^ ERROR mismatched types
        x_i128 > x_u16;
        //~^ ERROR mismatched types
        x_i128 > x_u32;
        //~^ ERROR mismatched types
        x_i128 > x_u64;
        //~^ ERROR mismatched types
        x_i128 > x_u128;
        //~^ ERROR mismatched types
        x_i128 > x_usize;
        //~^ ERROR mismatched types

        x_isize > x_u8;
        //~^ ERROR mismatched types
        x_isize > x_u16;
        //~^ ERROR mismatched types
        x_isize > x_u32;
        //~^ ERROR mismatched types
        x_isize > x_u64;
        //~^ ERROR mismatched types
        x_isize > x_u128;
        //~^ ERROR mismatched types
        x_isize > x_usize;
        //~^ ERROR mismatched types
    }
}
