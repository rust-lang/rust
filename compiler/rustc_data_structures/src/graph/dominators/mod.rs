//! Finding the dominators in a control-flow graph.
//!
//! Algorithm based on Loukas Georgiadis,
//! "Linear-Time Algorithms for Dominators and Related Problems",
//! ftp://ftp.cs.princeton.edu/techreports/2005/737.pdf

use super::iterate::reverse_post_order;
use super::ControlFlowGraph;
use rustc_index::vec::{Idx, IndexVec};
use std::cmp::Ordering;

#[cfg(test)]
mod tests;

pub fn dominators<G: ControlFlowGraph>(graph: G) -> Dominators<G::Node> {
    let start_node = graph.start_node();
    let rpo = reverse_post_order(&graph, start_node);
    dominators_given_rpo(graph, &rpo)
}

struct PreOrderFrame<Node, Iter> {
    node: Node,
    iter: Iter,
}

fn dominators_given_rpo<G: ControlFlowGraph>(graph: G, rpo: &[G::Node]) -> Dominators<G::Node> {
    let start_node = graph.start_node();
    assert_eq!(rpo[0], start_node);

    // compute the post order index (rank) for each node
    let mut post_order_rank = IndexVec::from_elem_n(0, graph.num_nodes());
    for (index, node) in rpo.iter().rev().cloned().enumerate() {
        post_order_rank[node] = index;
    }

    let mut visited = BitSet::new_empty(graph.num_nodes());
    let mut parent: IndexVec<G::Node, Option<G::Node>> =
        IndexVec::from_elem_n(None, graph.num_nodes());
    let mut pre_order_index: IndexVec<G::Node, Option<usize>> =
        IndexVec::from_elem_n(None, graph.num_nodes());
    let mut pre_order_nodes = Vec::with_capacity(rpo.len());

    let mut stack = vec![PreOrderFrame {
        node: graph.start_node(),
        iter: graph.successors(graph.start_node()),
    }];
    visited.insert(graph.start_node());
    let mut idx = 0;
    pre_order_index[graph.start_node()] = Some(0);
    idx += 1;
    pre_order_nodes.push(graph.start_node());

    'recurse: while let Some(frame) = stack.last_mut() {
        while let Some(successor) = frame.iter.next() {
            if visited.insert(successor) {
                parent[successor] = Some(frame.node);
                pre_order_index[successor] = Some(idx);
                pre_order_nodes.push(successor);
                idx += 1;

                stack.push(PreOrderFrame { node: successor, iter: graph.successors(successor) });
                continue 'recurse;
            }
        }
        stack.pop();
    }

    let mut ancestor = IndexVec::from_elem_n(None, graph.num_nodes());
    let mut idom = IndexVec::from_elem_n(graph.start_node(), graph.num_nodes());
    let mut semi = IndexVec::from_fn_n(std::convert::identity, graph.num_nodes());
    let mut label = semi.clone();
    let mut bucket = IndexVec::from_elem_n(vec![], graph.num_nodes());

    for &w in pre_order_nodes[1..].iter().rev() {
        semi[w] = w;
        for v in graph.predecessors(w) {
            let x = eval(&pre_order_index, &mut ancestor, &semi, &mut label, v);
            semi[w] = if pre_order_index[semi[w]].unwrap() < pre_order_index[semi[x]].unwrap() {
                semi[w]
            } else {
                semi[x]
            };
        }
        // semi[w] is now semidominator(w).

        bucket[semi[w]].push(w);

        link(&mut ancestor, &parent, w);
        let z = parent[w].unwrap();
        for v in std::mem::take(&mut bucket[z]) {
            let y = eval(&pre_order_index, &mut ancestor, &semi, &mut label, v);
            idom[v] = if pre_order_index[semi[y]] < pre_order_index[z] { y } else { z };
        }
    }
    for &w in pre_order_nodes.iter().skip(1) {
        if idom[w] != semi[w] {
            idom[w] = idom[idom[w]];
        }
    }

    let mut immediate_dominators = IndexVec::from_elem_n(None, graph.num_nodes());
    for (node, idom_slot) in immediate_dominators.iter_enumerated_mut() {
        if pre_order_index[node].is_some() {
            *idom_slot = Some(idom[node]);
        }
    }

    Dominators { post_order_rank, immediate_dominators }
}

fn eval<N: Idx>(
    pre_order_index: &IndexVec<N, Option<usize>>,
    ancestor: &mut IndexVec<N, Option<N>>,
    semi: &IndexVec<N, N>,
    label: &mut IndexVec<N, N>,
    node: N,
) -> N {
    if ancestor[node].is_some() {
        compress(pre_order_index, ancestor, semi, label, node);
        label[node]
    } else {
        node
    }
}

fn compress<N: Idx>(
    pre_order_index: &IndexVec<N, Option<usize>>,
    ancestor: &mut IndexVec<N, Option<N>>,
    semi: &IndexVec<N, N>,
    label: &mut IndexVec<N, N>,
    v: N,
) {
    let u = ancestor[v].unwrap();
    if ancestor[u].is_some() {
        compress(pre_order_index, ancestor, semi, label, u);
        if pre_order_index[semi[label[u]]] < pre_order_index[semi[label[v]]] {
            label[v] = label[u];
        }
        ancestor[v] = ancestor[u];
    }
}

fn link<N: Idx>(ancestor: &mut IndexVec<N, Option<N>>, parent: &IndexVec<N, Option<N>>, w: N) {
    ancestor[w] = Some(parent[w].unwrap());
}

#[derive(Clone, Debug)]
pub struct Dominators<N: Idx> {
    post_order_rank: IndexVec<N, usize>,
    immediate_dominators: IndexVec<N, Option<N>>,
}

impl<Node: Idx> Dominators<Node> {
    pub fn dummy() -> Self {
        Self { post_order_rank: IndexVec::new(), immediate_dominators: IndexVec::new() }
    }

    pub fn is_reachable(&self, node: Node) -> bool {
        self.immediate_dominators[node].is_some()
    }

    pub fn immediate_dominator(&self, node: Node) -> Node {
        assert!(self.is_reachable(node), "node {:?} is not reachable", node);
        self.immediate_dominators[node].unwrap()
    }

    pub fn dominators(&self, node: Node) -> Iter<'_, Node> {
        assert!(self.is_reachable(node), "node {:?} is not reachable", node);
        Iter { dominators: self, node: Some(node) }
    }

    pub fn is_dominated_by(&self, node: Node, dom: Node) -> bool {
        // FIXME -- could be optimized by using post-order-rank
        self.dominators(node).any(|n| n == dom)
    }

    /// Provide deterministic ordering of nodes such that, if any two nodes have a dominator
    /// relationship, the dominator will always precede the dominated. (The relative ordering
    /// of two unrelated nodes will also be consistent, but otherwise the order has no
    /// meaning.) This method cannot be used to determine if either Node dominates the other.
    pub fn rank_partial_cmp(&self, lhs: Node, rhs: Node) -> Option<Ordering> {
        self.post_order_rank[lhs].partial_cmp(&self.post_order_rank[rhs])
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
            Some(node)
        } else {
            None
        }
    }
}
