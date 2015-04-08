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
#![feature(box_syntax, collections)]

extern crate collections;

use std::collections::HashMap;

trait Graph<Node, Edge> {
    fn f(&self, Edge);
    fn g(&self, Node);

}

impl<E> Graph<isize, E> for HashMap<isize, isize> {
    fn f(&self, _e: E) {
        panic!();
    }
    fn g(&self, _e: isize) {
        panic!();
    }
}

pub fn main() {
    let g : Box<HashMap<isize,isize>> = box HashMap::new();
    let _g2 : Box<Graph<isize,isize>> = g as Box<Graph<isize,isize>>;
}
