// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::super::indexed_vec::IndexVec;
use super::{DirectedGraph, WithSuccessors, WithNumNodes};

#[cfg(test)]
mod test;

pub fn post_order_from<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
) -> Vec<G::Node> {
    post_order_from_to(graph, start_node, None)
}

pub fn post_order_from_to<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
    end_node: Option<G::Node>,
) -> Vec<G::Node> {
    let mut visited: IndexVec<G::Node, bool> = IndexVec::from_elem_n(false, graph.num_nodes());
    let mut result: Vec<G::Node> = Vec::with_capacity(graph.num_nodes());
    if let Some(end_node) = end_node {
        visited[end_node] = true;
    }
    post_order_walk(graph, start_node, &mut result, &mut visited);
    result
}

fn post_order_walk<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    node: G::Node,
    result: &mut Vec<G::Node>,
    visited: &mut IndexVec<G::Node, bool>,
) {
    if visited[node] {
        return;
    }
    visited[node] = true;

    for successor in graph.successors(node) {
        post_order_walk(graph, successor, result, visited);
    }

    result.push(node);
}

pub fn reverse_post_order<G: DirectedGraph + WithSuccessors + WithNumNodes>(
    graph: &G,
    start_node: G::Node,
) -> Vec<G::Node> {
    let mut vec = post_order_from(graph, start_node);
    vec.reverse();
    vec
}
