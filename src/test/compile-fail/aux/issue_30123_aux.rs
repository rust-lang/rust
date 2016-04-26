// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

pub struct Directed;
pub struct Undirected;

pub struct Graph<N, E, Ty = Directed> {
    nodes: Vec<PhantomData<N>>,
    edges: Vec<PhantomData<E>>,
    ty: PhantomData<Ty>,
}


impl<N, E> Graph<N, E, Directed> {
    pub fn new() -> Self {
        Graph{nodes: Vec::new(), edges: Vec::new(), ty: PhantomData}
    }
}

impl<N, E> Graph<N, E, Undirected> {
    pub fn new_undirected() -> Self {
        Graph{nodes: Vec::new(), edges: Vec::new(), ty: PhantomData}
    }
}
