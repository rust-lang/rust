// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

extern crate collections;

use std::collections::HashMap;

trait Graph<Node, Edge> {
    fn f(&self, Edge);
    fn g(&self, Node);

}

impl<E> Graph<int, E> for HashMap<int, int> {
    fn f(&self, _e: E) {
        panic!();
    }
    fn g(&self, _e: int) {
        panic!();
    }
}

pub fn main() {
    let g : Box<HashMap<int,int>> = box HashMap::new();
    let _g2 : Box<Graph<int,int>> = g as Box<Graph<int,int>>;
}
