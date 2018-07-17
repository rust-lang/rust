// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait ToPrimitive {
    fn to_int(&self) -> isize { 0 }
}

impl ToPrimitive for i32 {}
impl ToPrimitive for isize {}

trait Add {
    fn to_int(&self) -> isize;
    fn add_dynamic(&self, other: &Add) -> isize;
}

impl Add for isize {
    fn to_int(&self) -> isize { *self }
    fn add_dynamic(&self, other: &Add) -> isize {
        self.to_int() + other.to_int() //~ ERROR multiple applicable items in scope
    }
}

fn main() { }
