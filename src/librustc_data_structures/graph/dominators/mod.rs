//! Algorithm citation:
//! A Simple, Fast Dominance Algorithm.
//! Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
//! Rice Computer Science TS-06-33870
//! <https://www.cs.rice.edu/~keith/EMBED/dom.pdf>

use super::super::indexed_vec::{Idx, IndexVec};
use super::iterate::reverse_post_order;
use super::ControlFlowGraph;

#[cfg(test)]
mod test;

pub fn dominators<G: ControlFlowGraph>(graph: &G) -> Dominators<G::Node> {
    let start_node = graph.start_node();
    let rpo = reverse_post_order(graph, start_node);
    dominators_given_rpo(graph, &rpo)
}

pub fn dominators_given_rpo<G: ControlFlowGraph>(
    graph: &G,
    rpo: &[G::Node],
) -> Dominators<G::Node> {
    let start_node = graph.start_node();
    assert_eq!(rpo[0], start_node);

    // compute the post order index (rank) for each node
    let mut post_order_rank: IndexVec<G::Node, usize> =
        (0..graph.num_nodes()).map(|_| 0).collect();
    for (index, node) in rpo.iter().rev().cloned().enumerate() {
        post_order_rank[node] = index;
    }

    let mut immediate_dominators: IndexVec<G::Node, Option<G::Node>> =
        (0..graph.num_nodes()).map(|_| None).collect();
    immediate_dominators[start_node] = Some(start_node);

    let mut changed = true;
    while changed {
        changed = false;

        for &node in &rpo[1..] {
            let mut new_idom = None;
            for pred in graph.predecessors(node) {
                if immediate_dominators[pred].is_some() {
                    // (*)
                    // (*) dominators for `pred` have been calculated
                    new_idom = intersect_opt(
                        &post_order_rank,
                        &immediate_dominators,
                        new_idom,
                        Some(pred),
                    );
                }
            }

            if new_idom != immediate_dominators[node] {
                immediate_dominators[node] = new_idom;
                changed = true;
            }
        }
    }

    Dominators {
        post_order_rank,
        immediate_dominators,
    }
}

fn intersect_opt<Node: Idx>(
    post_order_rank: &IndexVec<Node, usize>,
    immediate_dominators: &IndexVec<Node, Option<Node>>,
    node1: Option<Node>,
    node2: Option<Node>,
) -> Option<Node> {
    match (node1, node2) {
        (None, None) => None,
        (Some(n), None) | (None, Some(n)) => Some(n),
        (Some(n1), Some(n2)) => Some(intersect(post_order_rank, immediate_dominators, n1, n2)),
    }
}

fn intersect<Node: Idx>(
    post_order_rank: &IndexVec<Node, usize>,
    immediate_dominators: &IndexVec<Node, Option<Node>>,
    mut node1: Node,
    mut node2: Node,
) -> Node {
    while node1 != node2 {
        while post_order_rank[node1] < post_order_rank[node2] {
            node1 = immediate_dominators[node1].unwrap();
        }

        while post_order_rank[node2] < post_order_rank[node1] {
            node2 = immediate_dominators[node2].unwrap();
        }
    }

    node1
}

#[derive(Clone, Debug)]
pub struct Dominators<N: Idx> {
    post_order_rank: IndexVec<N, usize>,
    immediate_dominators: IndexVec<N, Option<N>>,
}

impl<Node: Idx> Dominators<Node> {
    pub fn is_reachable(&self, node: Node) -> bool {
        self.immediate_dominators[node].is_some()
    }

    pub fn immediate_dominator(&self, node: Node) -> Node {
        assert!(self.is_reachable(node), "node {:?} is not reachable", node);
        self.immediate_dominators[node].unwrap()
    }

    pub fn dominators(&self, node: Node) -> Iter<'_, Node> {
        assert!(self.is_reachable(node), "node {:?} is not reachable", node);
        Iter {
            dominators: self,
            node: Some(node),
        }
    }

    pub fn is_dominated_by(&self, node: Node, dom: Node) -> bool {
        // FIXME -- could be optimized by using post-order-rank
        self.dominators(node).any(|n| n == dom)
    }

    #[cfg(test)]
    fn all_immediate_dominators(&self) -> &IndexVec<Node, Option<Node>> {
        &self.immediate_dominators
    }
}

pub struct Iter<'dom, Node: Idx> {
    dominators: &'dom Dominators<Node>,
    node: Option<Node>,
}

impl<'dom, Node: Idx> Iterator for Iter<'dom, Node> {
    type Item = Node;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.node {
            let dom = self.dominators.immediate_dominator(node);
            if dom == node {
                self.node = None; // reached the root
            } else {
                self.node = Some(dom);
            }
            return Some(node);
        } else {
            return None;
        }
    }
}
