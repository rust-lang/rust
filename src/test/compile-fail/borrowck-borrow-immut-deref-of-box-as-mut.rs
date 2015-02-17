// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

struct A;

impl A {
    fn foo(&mut self) {
    }
}

pub fn main() {
    let a: Box<_> = box A;
    a.foo();
    //~^ ERROR cannot borrow immutable `Box` content `*a` as mutable
}
