// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

use std::marker::PhantomData;

enum NodeContents<'a> {
    Children(Vec<Node<'a>>),
}

impl<'a> Drop for NodeContents<'a> {
    //~^ ERROR cannot implement a destructor on a structure with type parameters
    fn drop( &mut self ) {
    }
}

struct Node<'a> {
    contents: NodeContents<'a>,
    marker: PhantomData<&'a ()>,
}

impl<'a> Node<'a> {
    fn noName(contents: NodeContents<'a>) -> Node<'a> {
        Node { contents: contents, marker: PhantomData }
    }
}

fn main() {}
