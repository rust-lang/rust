#![allow(deprecated, deprecated_in_future)] // can be removed if different fns are chosen
// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let x: &'static u8 = &u8::max_value();
    let x: &'static u16 = &u16::max_value();
    let x: &'static u32 = &u32::max_value();
    let x: &'static u64 = &u64::max_value();
    let x: &'static u128 = &u128::max_value();
    let x: &'static usize = &usize::max_value();
    let x: &'static u8 = &u8::min_value();
    let x: &'static u16 = &u16::min_value();
    let x: &'static u32 = &u32::min_value();
    let x: &'static u64 = &u64::min_value();
    let x: &'static u128 = &u128::min_value();
    let x: &'static usize = &usize::min_value();
    let x: &'static i8 = &i8::max_value();
    let x: &'static i16 = &i16::max_value();
    let x: &'static i32 = &i32::max_value();
    let x: &'static i64 = &i64::max_value();
    let x: &'static i128 = &i128::max_value();
    let x: &'static isize = &isize::max_value();
    let x: &'static i8 = &i8::min_value();
    let x: &'static i16 = &i16::min_value();
    let x: &'static i32 = &i32::min_value();
    let x: &'static i64 = &i64::min_value();
    let x: &'static i128 = &i128::min_value();
    let x: &'static isize = &isize::min_value();
}
