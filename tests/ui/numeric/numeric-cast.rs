//@ run-rustfix

// The `try_into` suggestion doesn't include this, but we do suggest it after applying it
use std::convert::TryInto;

fn foo<N>(_x: N) {}

fn main() {
    let x_usize: usize = 1;
    let x_u64: u64 = 2;
    let x_u32: u32 = 3;
    let x_u16: u16 = 4;
    let x_u8: u8 = 5;
    let x_isize: isize = 6;
    let x_i64: i64 = 7;
    let x_i32: i32 = 8;
    let x_i16: i16 = 9;
    let x_i8: i8 = 10;
    let x_f64: f64 = 11.0;
    let x_f32: f32 = 12.0;

    foo::<usize>(x_usize);
    foo::<usize>(x_u64);
    //~^ ERROR mismatched types
    foo::<usize>(x_u32);
    //~^ ERROR mismatched types
    foo::<usize>(x_u16);
    //~^ ERROR mismatched types
    foo::<usize>(x_u8);
    //~^ ERROR mismatched types
    foo::<usize>(x_isize);
    //~^ ERROR mismatched types
    foo::<usize>(x_i64);
    //~^ ERROR mismatched types
    foo::<usize>(x_i32);
    //~^ ERROR mismatched types
    foo::<usize>(x_i16);
    //~^ ERROR mismatched types
    foo::<usize>(x_i8);
    //~^ ERROR mismatched types
    // foo::<usize>(x_f64);
    // foo::<usize>(x_f32);

    foo::<isize>(x_usize);
    //~^ ERROR mismatched types
    foo::<isize>(x_u64);
    //~^ ERROR mismatched types
    foo::<isize>(x_u32);
    //~^ ERROR mismatched types
    foo::<isize>(x_u16);
    //~^ ERROR mismatched types
    foo::<isize>(x_u8);
    //~^ ERROR mismatched types
    foo::<isize>(x_isize);
    foo::<isize>(x_i64);
    //~^ ERROR mismatched types
    foo::<isize>(x_i32);
    //~^ ERROR mismatched types
    foo::<isize>(x_i16);
    //~^ ERROR mismatched types
    foo::<isize>(x_i8);
    //~^ ERROR mismatched types
    // foo::<isize>(x_f64);
    // foo::<isize>(x_f32);

    foo::<u64>(x_usize);
    //~^ ERROR mismatched types
    foo::<u64>(x_u64);
    foo::<u64>(x_u32);
    //~^ ERROR mismatched types
    foo::<u64>(x_u16);
    //~^ ERROR mismatched types
    foo::<u64>(x_u8);
    //~^ ERROR mismatched types
    foo::<u64>(x_isize);
    //~^ ERROR mismatched types
    foo::<u64>(x_i64);
    //~^ ERROR mismatched types
    foo::<u64>(x_i32);
    //~^ ERROR mismatched types
    foo::<u64>(x_i16);
    //~^ ERROR mismatched types
    foo::<u64>(x_i8);
    //~^ ERROR mismatched types
    // foo::<u64>(x_f64);
    // foo::<u64>(x_f32);

    foo::<i64>(x_usize);
    //~^ ERROR mismatched types
    foo::<i64>(x_u64);
    //~^ ERROR mismatched types
    foo::<i64>(x_u32);
    //~^ ERROR mismatched types
    foo::<i64>(x_u16);
    //~^ ERROR mismatched types
    foo::<i64>(x_u8);
    //~^ ERROR mismatched types
    foo::<i64>(x_isize);
    //~^ ERROR mismatched types
    foo::<i64>(x_i64);
    foo::<i64>(x_i32);
    //~^ ERROR mismatched types
    foo::<i64>(x_i16);
    //~^ ERROR mismatched types
    foo::<i64>(x_i8);
    //~^ ERROR mismatched types
    // foo::<i64>(x_f64);
    // foo::<i64>(x_f32);

    foo::<u32>(x_usize);
    //~^ ERROR mismatched types
    foo::<u32>(x_u64);
    //~^ ERROR mismatched types
    foo::<u32>(x_u32);
    foo::<u32>(x_u16);
    //~^ ERROR mismatched types
    foo::<u32>(x_u8);
    //~^ ERROR mismatched types
    foo::<u32>(x_isize);
    //~^ ERROR mismatched types
    foo::<u32>(x_i64);
    //~^ ERROR mismatched types
    foo::<u32>(x_i32);
    //~^ ERROR mismatched types
    foo::<u32>(x_i16);
    //~^ ERROR mismatched types
    foo::<u32>(x_i8);
    //~^ ERROR mismatched types
    // foo::<u32>(x_f64);
    // foo::<u32>(x_f32);

    foo::<i32>(x_usize);
    //~^ ERROR mismatched types
    foo::<i32>(x_u64);
    //~^ ERROR mismatched types
    foo::<i32>(x_u32);
    //~^ ERROR mismatched types
    foo::<i32>(x_u16);
    //~^ ERROR mismatched types
    foo::<i32>(x_u8);
    //~^ ERROR mismatched types
    foo::<i32>(x_isize);
    //~^ ERROR mismatched types
    foo::<i32>(x_i64);
    //~^ ERROR mismatched types
    foo::<i32>(x_i32);
    foo::<i32>(x_i16);
    //~^ ERROR mismatched types
    foo::<i32>(x_i8);
    //~^ ERROR mismatched types
    // foo::<i32>(x_f64);
    // foo::<i32>(x_f32);

    foo::<u16>(x_usize);
    //~^ ERROR mismatched types
    foo::<u16>(x_u64);
    //~^ ERROR mismatched types
    foo::<u16>(x_u32);
    //~^ ERROR mismatched types
    foo::<u16>(x_u16);
    foo::<u16>(x_u8);
    //~^ ERROR mismatched types
    foo::<u16>(x_isize);
    //~^ ERROR mismatched types
    foo::<u16>(x_i64);
    //~^ ERROR mismatched types
    foo::<u16>(x_i32);
    //~^ ERROR mismatched types
    foo::<u16>(x_i16);
    //~^ ERROR mismatched types
    foo::<u16>(x_i8);
    //~^ ERROR mismatched types
    // foo::<u16>(x_f64);
    // foo::<u16>(x_f32);

    foo::<i16>(x_usize);
    //~^ ERROR mismatched types
    foo::<i16>(x_u64);
    //~^ ERROR mismatched types
    foo::<i16>(x_u32);
    //~^ ERROR mismatched types
    foo::<i16>(x_u16);
    //~^ ERROR mismatched types
    foo::<i16>(x_u8);
    //~^ ERROR mismatched types
    foo::<i16>(x_isize);
    //~^ ERROR mismatched types
    foo::<i16>(x_i64);
    //~^ ERROR mismatched types
    foo::<i16>(x_i32);
    //~^ ERROR mismatched types
    foo::<i16>(x_i16);
    foo::<i16>(x_i8);
    //~^ ERROR mismatched types
    // foo::<i16>(x_f64);
    // foo::<i16>(x_f32);

    foo::<u8>(x_usize);
    //~^ ERROR mismatched types
    foo::<u8>(x_u64);
    //~^ ERROR mismatched types
    foo::<u8>(x_u32);
    //~^ ERROR mismatched types
    foo::<u8>(x_u16);
    //~^ ERROR mismatched types
    foo::<u8>(x_u8);
    foo::<u8>(x_isize);
    //~^ ERROR mismatched types
    foo::<u8>(x_i64);
    //~^ ERROR mismatched types
    foo::<u8>(x_i32);
    //~^ ERROR mismatched types
    foo::<u8>(x_i16);
    //~^ ERROR mismatched types
    foo::<u8>(x_i8);
    //~^ ERROR mismatched types
    // foo::<u8>(x_f64);
    // foo::<u8>(x_f32);

    foo::<i8>(x_usize);
    //~^ ERROR mismatched types
    foo::<i8>(x_u64);
    //~^ ERROR mismatched types
    foo::<i8>(x_u32);
    //~^ ERROR mismatched types
    foo::<i8>(x_u16);
    //~^ ERROR mismatched types
    foo::<i8>(x_u8);
    //~^ ERROR mismatched types
    foo::<i8>(x_isize);
    //~^ ERROR mismatched types
    foo::<i8>(x_i64);
    //~^ ERROR mismatched types
    foo::<i8>(x_i32);
    //~^ ERROR mismatched types
    foo::<i8>(x_i16);
    //~^ ERROR mismatched types
    foo::<i8>(x_i8);
    // foo::<i8>(x_f64);
    // foo::<i8>(x_f32);

    foo::<f64>(x_usize);
    //~^ ERROR mismatched types
    foo::<f64>(x_u64);
    //~^ ERROR mismatched types
    foo::<f64>(x_u32);
    //~^ ERROR mismatched types
    foo::<f64>(x_u16);
    //~^ ERROR mismatched types
    foo::<f64>(x_u8);
    //~^ ERROR mismatched types
    foo::<f64>(x_isize);
    //~^ ERROR mismatched types
    foo::<f64>(x_i64);
    //~^ ERROR mismatched types
    foo::<f64>(x_i32);
    //~^ ERROR mismatched types
    foo::<f64>(x_i16);
    //~^ ERROR mismatched types
    foo::<f64>(x_i8);
    //~^ ERROR mismatched types
    foo::<f64>(x_f64);
    foo::<f64>(x_f32);
    //~^ ERROR mismatched types

    foo::<f32>(x_usize);
    //~^ ERROR mismatched types
    foo::<f32>(x_u64);
    //~^ ERROR mismatched types
    foo::<f32>(x_u32);
    //~^ ERROR mismatched types
    foo::<f32>(x_u16);
    //~^ ERROR mismatched types
    foo::<f32>(x_u8);
    //~^ ERROR mismatched types
    foo::<f32>(x_isize);
    //~^ ERROR mismatched types
    foo::<f32>(x_i64);
    //~^ ERROR mismatched types
    foo::<f32>(x_i32);
    //~^ ERROR mismatched types
    foo::<f32>(x_i16);
    //~^ ERROR mismatched types
    foo::<f32>(x_i8);
    //~^ ERROR mismatched types
    // foo::<f32>(x_f64);
    foo::<f32>(x_f32);

    foo::<u32>(x_u8 as u16);
    //~^ ERROR mismatched types
    foo::<i32>(-x_i8);
    //~^ ERROR mismatched types
}
