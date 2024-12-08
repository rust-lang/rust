#![allow(
    clippy::cast_lossless,
    clippy::legacy_numeric_constants,
    unused,
    // Int::max_value will be deprecated in the future
    deprecated,
)]
#![warn(clippy::checked_conversions)]

// Positive tests

// Signed to unsigned

pub fn i64_to_u32(value: i64) {
    let _ = value <= (u32::max_value() as i64) && value >= 0;
    let _ = value <= (u32::MAX as i64) && value >= 0;
}

pub fn i64_to_u16(value: i64) {
    let _ = value <= i64::from(u16::max_value()) && value >= 0;
    let _ = value <= i64::from(u16::MAX) && value >= 0;
}

pub fn isize_to_u8(value: isize) {
    let _ = value <= (u8::max_value() as isize) && value >= 0;
    let _ = value <= (u8::MAX as isize) && value >= 0;
}

// Signed to signed

pub fn i64_to_i32(value: i64) {
    let _ = value <= (i32::max_value() as i64) && value >= (i32::min_value() as i64);
    let _ = value <= (i32::MAX as i64) && value >= (i32::MIN as i64);
}

pub fn i64_to_i16(value: i64) {
    let _ = value <= i64::from(i16::max_value()) && value >= i64::from(i16::min_value());
    let _ = value <= i64::from(i16::MAX) && value >= i64::from(i16::MIN);
}

// Unsigned to X

pub fn u32_to_i32(value: u32) {
    let _ = value <= i32::max_value() as u32;
    let _ = value <= i32::MAX as u32;
}

pub fn usize_to_isize(value: usize) {
    let _ = value <= isize::max_value() as usize && value as i32 == 5;
    let _ = value <= isize::MAX as usize && value as i32 == 5;
}

pub fn u32_to_u16(value: u32) {
    let _ = value <= u16::max_value() as u32 && value as i32 == 5;
    let _ = value <= u16::MAX as u32 && value as i32 == 5;
}

// Negative tests

pub fn no_i64_to_i32(value: i64) {
    let _ = value <= (i32::max_value() as i64) && value >= 0;
    let _ = value <= (i32::MAX as i64) && value >= 0;
}

pub fn no_isize_to_u8(value: isize) {
    let _ = value <= (u8::max_value() as isize) && value >= (u8::min_value() as isize);
    let _ = value <= (u8::MAX as isize) && value >= (u8::MIN as isize);
}

pub fn i8_to_u8(value: i8) {
    let _ = value >= 0;
}

// Do not lint
pub const fn issue_8898(i: u32) -> bool {
    i <= i32::MAX as u32
}

#[clippy::msrv = "1.33"]
fn msrv_1_33() {
    let value: i64 = 33;
    let _ = value <= (u32::MAX as i64) && value >= 0;
}

#[clippy::msrv = "1.34"]
fn msrv_1_34() {
    let value: i64 = 34;
    let _ = value <= (u32::MAX as i64) && value >= 0;
}

fn main() {}
