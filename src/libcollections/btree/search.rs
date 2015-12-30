// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::Ordering;

use borrow::Borrow;

use super::node::{Handle, NodeRef, marker};

use super::node::ForceResult::*;
use self::SearchResult::*;

pub enum SearchResult<Lifetime, K, V, FoundType, GoDownType> {
    Found(Handle<NodeRef<Lifetime, K, V, FoundType>, marker::KV>),
    GoDown(Handle<NodeRef<Lifetime, K, V, GoDownType>, marker::Edge>)
}

pub fn search_tree<Lifetime, K, V, Q: ?Sized>(
    mut node: NodeRef<Lifetime, K, V, marker::LeafOrInternal>,
    key: &Q
) -> SearchResult<Lifetime, K, V, marker::LeafOrInternal, marker::Leaf>
        where Q: Ord, K: Borrow<Q> {

    loop {
        match search_node(node, key) {
            Found(handle) => return Found(handle),
            GoDown(handle) => match handle.force() {
                Leaf(leaf) => return GoDown(leaf),
                Internal(internal) => {
                    node = internal.descend();
                    continue;
                }
            }
        }
    }
}

pub fn search_node<Lifetime, K, V, Type, Q: ?Sized>(
    node: NodeRef<Lifetime, K, V, Type>,
    key: &Q
) -> SearchResult<Lifetime, K, V, Type, Type>
        where Q: Ord, K: Borrow<Q> {

    match search_linear(&node, key) {
        (idx, true) => Found(
            unsafe { Handle::new(node, idx) }
        ),
        (idx, false) => SearchResult::GoDown(
            unsafe { Handle::new(node, idx) }
        )
    }
}

fn search_linear<Lifetime, K, V, Type, Q: ?Sized>(
    node: &NodeRef<Lifetime, K, V, Type>,
    key: &Q
) -> (usize, bool)
        where Q: Ord, K: Borrow<Q> {

    for (i, k) in node.keys().iter().enumerate() {
        match key.cmp(k.borrow()) {
            Ordering::Greater => {},
            Ordering::Equal => return (i, true),
            Ordering::Less => return (i, false)
        }
    }
    (node.keys().len(), false)
}

