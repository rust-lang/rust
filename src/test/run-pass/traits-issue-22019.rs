// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test an issue where global caching was causing free regions from
// distinct scopes to be compared (`'g` and `'h`). The only important
// thing is that compilation succeeds here.

#![allow(missing_copy_implementations)]
#![allow(unused_variables)]

use std::borrow::ToOwned;

pub struct CFGNode;

pub type Node<'a> = &'a CFGNode;

pub trait GraphWalk<'c, N> {
    /// Returns all the nodes in this graph.
    fn nodes(&'c self) where [N]:ToOwned<Owned=Vec<N>>;
}

impl<'g> GraphWalk<'g, Node<'g>> for u32
{
    fn nodes(&'g self) where [Node<'g>]:ToOwned<Owned=Vec<Node<'g>>>
    { loop { } }
}

impl<'h> GraphWalk<'h, Node<'h>> for u64
{
    fn nodes(&'h self) where [Node<'h>]:ToOwned<Owned=Vec<Node<'h>>>
    { loop { } }
}

fn main()  { }
