// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::{Gc, GC};

fn box_it<T:'static>(x: Box<T>) -> Gc<Box<T>> { return box(GC) x; }

struct Box<T> {x: T, y: T, z: T}

pub fn main() {
    let x: Gc<Box<int>> = box_it::<int>(Box{x: 1, y: 2, z: 3});
    assert_eq!(x.y, 2);
}
