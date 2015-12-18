// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

pub trait NumCast: Sized {
    fn from(i: i32) -> Option<Self>;
}

pub trait NumExt: PartialEq + NumCast {}

impl NumExt for f32 {}
impl NumExt for isize {}

impl NumCast for f32 {
    fn from(i: i32) -> Option<f32> { Some(i as f32) }
}
impl NumCast for isize {
    fn from(i: i32) -> Option<isize> { Some(i as isize) }
}

fn num_eq_one<T:NumExt>() -> T {
    NumCast::from(1).unwrap()
}

pub fn main() {
    num_eq_one::<isize>(); // you need to actually use the function to trigger the ICE
}
