// run-rustfix

#![warn(clippy::checked_conversions)]
#![allow(clippy::cast_lossless)]
#![allow(dead_code)]
use std::convert::TryFrom;

// Positive tests

// Signed to unsigned

fn i64_to_u32(value: i64) -> Option<u32> {
    if value <= (u32::max_value() as i64) && value >= 0 {
        Some(value as u32)
    } else {
        None
    }
}

fn i64_to_u16(value: i64) -> Option<u16> {
    if value <= i64::from(u16::max_value()) && value >= 0 {
        Some(value as u16)
    } else {
        None
    }
}

fn isize_to_u8(value: isize) -> Option<u8> {
    if value <= (u8::max_value() as isize) && value >= 0 {
        Some(value as u8)
    } else {
        None
    }
}

// Signed to signed

fn i64_to_i32(value: i64) -> Option<i32> {
    if value <= (i32::max_value() as i64) && value >= (i32::min_value() as i64) {
        Some(value as i32)
    } else {
        None
    }
}

fn i64_to_i16(value: i64) -> Option<i16> {
    if value <= i64::from(i16::max_value()) && value >= i64::from(i16::min_value()) {
        Some(value as i16)
    } else {
        None
    }
}

// Unsigned to X

fn u32_to_i32(value: u32) -> Option<i32> {
    if value <= i32::max_value() as u32 {
        Some(value as i32)
    } else {
        None
    }
}

fn usize_to_isize(value: usize) -> isize {
    if value <= isize::max_value() as usize && value as i32 == 5 {
        5
    } else {
        1
    }
}

fn u32_to_u16(value: u32) -> isize {
    if value <= u16::max_value() as u32 && value as i32 == 5 {
        5
    } else {
        1
    }
}

// Negative tests

fn no_i64_to_i32(value: i64) -> Option<i32> {
    if value <= (i32::max_value() as i64) && value >= 0 {
        Some(value as i32)
    } else {
        None
    }
}

fn no_isize_to_u8(value: isize) -> Option<u8> {
    if value <= (u8::max_value() as isize) && value >= (u8::min_value() as isize) {
        Some(value as u8)
    } else {
        None
    }
}

fn i8_to_u8(value: i8) -> Option<u8> {
    if value >= 0 {
        Some(value as u8)
    } else {
        None
    }
}

fn main() {}
