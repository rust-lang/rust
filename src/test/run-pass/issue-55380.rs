// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(specialization)]

pub trait Foo {
    fn abc() -> u32;
    fn def() -> u32;
}

pub trait Marker {}

impl Marker for () {}

impl<T> Foo for T {
    default fn abc() -> u32 { 16 }
    default fn def() -> u32 { 42 }
}

impl<T: Marker> Foo for T {
    fn def() -> u32 {
        Self::abc()
    }
}

fn main() {
   assert_eq!(<()>::def(), 16);
   assert_eq!(<i32>::def(), 42);
}
