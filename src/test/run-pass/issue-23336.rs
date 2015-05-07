// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Data { fn doit(&self) {} }
impl<T> Data for T {}
pub trait UnaryLogic { type D: Data; }
impl UnaryLogic for () { type D = i32; }

pub fn crashes<T: UnaryLogic>(t: T::D) {
    t.doit();
}

fn main() { crashes::<()>(0); }
