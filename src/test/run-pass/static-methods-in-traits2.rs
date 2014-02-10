// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Number: NumConv {
    fn from<T:Number>(n: T) -> Self;
}

impl Number for f64 {
    fn from<T:Number>(n: T) -> f64 { n.to_float() }
}

pub trait NumConv {
    fn to_float(&self) -> f64;
}

impl NumConv for f64 {
    fn to_float(&self) -> f64 { *self }
}

pub fn main() {
    let _: f64 = Number::from(0.0f64);
}
