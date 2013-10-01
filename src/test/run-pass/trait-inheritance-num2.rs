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

extern mod extra;

use std::cmp::{Eq, Ord};

pub trait TypeExt {}


impl TypeExt for u8 {}
impl TypeExt for u16 {}
impl TypeExt for u32 {}
impl TypeExt for u64 {}
impl TypeExt for uint {}

impl TypeExt for i8 {}
impl TypeExt for i16 {}
impl TypeExt for i32 {}
impl TypeExt for i64 {}
impl TypeExt for int {}

impl TypeExt for f32 {}
impl TypeExt for f64 {}


pub trait NumExt: TypeExt + Eq + Ord + Num + NumCast {}

impl NumExt for u8 {}
impl NumExt for u16 {}
impl NumExt for u32 {}
impl NumExt for u64 {}
impl NumExt for uint {}

impl NumExt for i8 {}
impl NumExt for i16 {}
impl NumExt for i32 {}
impl NumExt for i64 {}
impl NumExt for int {}

impl NumExt for f32 {}
impl NumExt for f64 {}


pub trait UnSignedExt: NumExt {}

impl UnSignedExt for u8 {}
impl UnSignedExt for u16 {}
impl UnSignedExt for u32 {}
impl UnSignedExt for u64 {}
impl UnSignedExt for uint {}


pub trait SignedExt: NumExt {}

impl SignedExt for i8 {}
impl SignedExt for i16 {}
impl SignedExt for i32 {}
impl SignedExt for i64 {}
impl SignedExt for int {}

impl SignedExt for f32 {}
impl SignedExt for f64 {}


pub trait IntegerExt: NumExt {}

impl IntegerExt for u8 {}
impl IntegerExt for u16 {}
impl IntegerExt for u32 {}
impl IntegerExt for u64 {}
impl IntegerExt for uint {}

impl IntegerExt for i8 {}
impl IntegerExt for i16 {}
impl IntegerExt for i32 {}
impl IntegerExt for i64 {}
impl IntegerExt for int {}


pub trait FloatExt: NumExt + ApproxEq<Self> {}

impl FloatExt for f32 {}
impl FloatExt for f64 {}


fn test_float_ext<T:FloatExt>(n: T) { println!("{}", n < n) }

pub fn main() {
    test_float_ext(1f32);
}
