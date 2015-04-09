// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to resolve conditional dispatch.  Here, the
// blanket impl for T:Copy coexists with an impl for Box<T>, because
// Box does not impl Copy.


#![allow(unknown_features)]
#![feature(box_syntax)]

trait Get {
    fn get(&self) -> Self;
}

trait MyCopy { fn copy(&self) -> Self; }
impl MyCopy for u16 { fn copy(&self) -> Self { *self } }
impl MyCopy for u32 { fn copy(&self) -> Self { *self } }
impl MyCopy for i32 { fn copy(&self) -> Self { *self } }
impl<T:Copy> MyCopy for Option<T> { fn copy(&self) -> Self { *self } }

impl<T:MyCopy> Get for T {
    fn get(&self) -> T { self.copy() }
}

impl Get for Box<i32> {
    fn get(&self) -> Box<i32> { box get_it(&**self) }
}

fn get_it<T:Get>(t: &T) -> T {
    (*t).get()
}

fn main() {
    assert_eq!(get_it(&1_u32), 1_u32);
    assert_eq!(get_it(&1_u16), 1_u16);
    assert_eq!(get_it(&Some(1_u16)), Some(1_u16));
    assert_eq!(get_it(&Box::new(1)), Box::new(1));
}
