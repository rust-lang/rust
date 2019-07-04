use super::super::indexed_vec::IndexVec;
use super::{DirectedGraph, WithNumNodes, WithSuccessors};
use crate::bit_set::BitSet;

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

/// A "depth-first search" iterator for a directed graph.
pub struct DepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    graph: &'graph G,
    stack: Vec<G::Node>,
    visited: BitSet<G::Node>,
}

impl<G> DepthFirstSearch<'graph, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    pub fn new(graph: &'graph G, start_node: G::Node) -> Self {
        Self { graph, stack: vec![start_node], visited: BitSet::new_empty(graph.num_nodes()) }
    }
}

impl<G> Iterator for DepthFirstSearch<'_, G>
where
    G: ?Sized + DirectedGraph + WithNumNodes + WithSuccessors,
{
    type Item = G::Node;

    fn next(&mut self) -> Option<G::Node> {
        let DepthFirstSearch { stack, visited, graph } = self;
        let n = stack.pop()?;
        stack.extend(graph.successors(n).filter(|&m| visited.insert(m)));
        Some(n)
    }
}
