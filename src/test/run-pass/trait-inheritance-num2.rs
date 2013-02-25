// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A more complex example of numeric extensions

use cmp::{Eq, Ord};
use num::NumCast::from;

extern mod std;
use std::cmp::FuzzyEq;

pub trait TypeExt {}


pub impl TypeExt for u8 {}
pub impl TypeExt for u16 {}
pub impl TypeExt for u32 {}
pub impl TypeExt for u64 {}
pub impl TypeExt for uint {}

pub impl TypeExt for i8 {}
pub impl TypeExt for i16 {}
pub impl TypeExt for i32 {}
pub impl TypeExt for i64 {}
pub impl TypeExt for int {}

pub impl TypeExt for f32 {}
pub impl TypeExt for f64 {}
pub impl TypeExt for float {}


pub trait NumExt: TypeExt Eq Ord NumCast {}

pub impl NumExt for u8 {}
pub impl NumExt for u16 {}
pub impl NumExt for u32 {}
pub impl NumExt for u64 {}
pub impl NumExt for uint {}

pub impl NumExt for i8 {}
pub impl NumExt for i16 {}
pub impl NumExt for i32 {}
pub impl NumExt for i64 {}
pub impl NumExt for int {}

pub impl NumExt for f32 {}
pub impl NumExt for f64 {}
pub impl NumExt for float {}


pub trait UnSignedExt: NumExt {}

pub impl UnSignedExt for u8 {}
pub impl UnSignedExt for u16 {}
pub impl UnSignedExt for u32 {}
pub impl UnSignedExt for u64 {}
pub impl UnSignedExt for uint {}


pub trait SignedExt: NumExt {}

pub impl SignedExt for i8 {}
pub impl SignedExt for i16 {}
pub impl SignedExt for i32 {}
pub impl SignedExt for i64 {}
pub impl SignedExt for int {}

pub impl SignedExt for f32 {}
pub impl SignedExt for f64 {}
pub impl SignedExt for float {}


pub trait IntegerExt: NumExt {}

pub impl IntegerExt for u8 {}
pub impl IntegerExt for u16 {}
pub impl IntegerExt for u32 {}
pub impl IntegerExt for u64 {}
pub impl IntegerExt for uint {}

pub impl IntegerExt for i8 {}
pub impl IntegerExt for i16 {}
pub impl IntegerExt for i32 {}
pub impl IntegerExt for i64 {}
pub impl IntegerExt for int {}


pub trait FloatExt: NumExt FuzzyEq<Self> {}

pub impl FloatExt for f32 {}
pub impl FloatExt for f64 {}
pub impl FloatExt for float {}


fn test_float_ext<T:FloatExt>(n: T) { io::println(fmt!("%?", n < n)) }

pub fn main() {
    test_float_ext(1f32);
}
