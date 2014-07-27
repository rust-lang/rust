// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate debug;

use std::gc::{GC, Gc};

fn magic(x: A) { println!("{:?}", x); }
fn magic2(x: Gc<int>) { println!("{:?}", x); }

struct A { a: Gc<int> }

pub fn main() {
    let a = A {a: box(GC) 10};
    let b = box(GC) 10;
    magic(a); magic(A {a: box(GC) 20});
    magic2(b); magic2(box(GC) 20);
}
