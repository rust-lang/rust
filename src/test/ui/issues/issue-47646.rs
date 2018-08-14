// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]
#![feature(nll)]

use std::collections::BinaryHeap;

fn main() {
    let mut heap: BinaryHeap<i32> = BinaryHeap::new();
    let borrow = heap.peek_mut();

    match (borrow, ()) {
        (Some(_), ()) => {
            println!("{:?}", heap); //~ ERROR cannot borrow `heap` as immutable
        }
        _ => {}
    };
}
