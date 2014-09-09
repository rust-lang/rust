// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let mut vector = vec![1u, 2];
    for &x in vector.iter() {
        let cap = vector.capacity();
        vector.grow(cap, &0u);      //~ ERROR cannot borrow
        *vector.get_mut(1u) = 5u;   //~ ERROR cannot borrow
    }
}

