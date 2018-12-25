// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

struct Node_ {
    a: Box<Cycle>
}

enum Cycle {
    Node(Node_),
    Empty,
}
fn main() {
    let mut x: Box<_> = box Cycle::Node(Node_ {a: box Cycle::Empty});
    // Create a cycle!
    match *x {
      Cycle::Node(ref mut y) => {
        y.a = x; //~ ERROR cannot move out of
      }
      Cycle::Empty => {}
    };
}
