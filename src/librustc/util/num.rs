// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait ToPrimitive {
    fn to_i8(&self) -> Option<i8>;
    fn to_i16(&self) -> Option<i16>;
    fn to_i32(&self) -> Option<i32>;
    fn to_i64(&self) -> Option<i64>;
    fn to_u8(&self) -> Option<u8>;
    fn to_u16(&self) -> Option<u16>;
    fn to_u32(&self) -> Option<u32>;
    fn to_u64(&self) -> Option<u64>;
}

impl ToPrimitive for i64 {
    fn to_i8(&self) -> Option<i8> {
        if *self < i8::min_value() as i64 || *self > i8::max_value() as i64 {
            None
        } else {
            Some(*self as i8)
        }
    }
    fn to_i16(&self) -> Option<i16> {
        if *self < i16::min_value() as i64 || *self > i16::max_value() as i64 {
            None
        } else {
            Some(*self as i16)
        }
    }
    fn to_i32(&self) -> Option<i32> {
        if *self < i32::min_value() as i64 || *self > i32::max_value() as i64 {
            None
        } else {
            Some(*self as i32)
        }
    }
    fn to_i64(&self) -> Option<i64> {
        Some(*self)
    }
    fn to_u8(&self) -> Option<u8> {
        if *self < 0 || *self > u8::max_value() as i64 {
            None
        } else {
            Some(*self as u8)
        }
    }
    fn to_u16(&self) -> Option<u16> {
        if *self < 0 || *self > u16::max_value() as i64 {
            None
        } else {
            Some(*self as u16)
        }
    }
    fn to_u32(&self) -> Option<u32> {
        if *self < 0 || *self > u32::max_value() as i64 {
            None
        } else {
            Some(*self as u32)
        }
    }
    fn to_u64(&self) -> Option<u64> {
        if *self < 0 {None} else {Some(*self as u64)}
    }
}

impl ToPrimitive for u64 {
    fn to_i8(&self) -> Option<i8> {
        if *self > i8::max_value() as u64 {None} else {Some(*self as i8)}
    }
    fn to_i16(&self) -> Option<i16> {
        if *self > i16::max_value() as u64 {None} else {Some(*self as i16)}
    }
    fn to_i32(&self) -> Option<i32> {
        if *self > i32::max_value() as u64 {None} else {Some(*self as i32)}
    }
    fn to_i64(&self) -> Option<i64> {
        if *self > i64::max_value() as u64 {None} else {Some(*self as i64)}
    }
    fn to_u8(&self) -> Option<u8> {
        if *self > u8::max_value() as u64 {None} else {Some(*self as u8)}
    }
    fn to_u16(&self) -> Option<u16> {
        if *self > u16::max_value() as u64 {None} else {Some(*self as u16)}
    }
    fn to_u32(&self) -> Option<u32> {
        if *self > u32::max_value() as u64 {None} else {Some(*self as u32)}
    }
    fn to_u64(&self) -> Option<u64> {
        Some(*self)
    }
}
