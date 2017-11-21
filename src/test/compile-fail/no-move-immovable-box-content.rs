// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(immovable_types)]

use std::marker::Immovable;

fn new_box_immovable() -> Box<Immovable> {
    // FIXME: we still can't create boxes of immovable types using
    // `Box::new`, please change this to use `Box::new` when that is possible
    panic!("please change this to use Box::new when that can be used")
}

fn move_from_box(b: Box<Immovable>) {
    *b; //~ ERROR cannot move immovable value out of a Box type
}

fn main() {
    let a = new_box_immovable();
    &*a; // ok
    move_from_box(a); // ok
}