// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(as_cell)]

use std::cell::Cell;

fn main() {
    let slice: &mut [i32] = &mut [1, 2, 3];
    let cell_slice: &Cell<[i32]> = Cell::from_mut(slice);
    assert_eq!(cell_slice[..].len(), 3);

    let sub_slice: &[Cell<i32>] = &cell_slice[1..];
    assert_eq!(sub_slice.len(), 2);
}
