//! Finding the dominators in a control-flow graph.
//!
//! Algorithm based on Loukas Georgiadis,
//! "Linear-Time Algorithms for Dominators and Related Problems",
//! ftp://ftp.cs.princeton.edu/techreports/2005/737.pdf

use super::ControlFlowGraph;
use rustc_index::vec::{Idx, IndexVec};
use std::cmp::Ordering;

#[cfg(test)]
mod tests;

struct PreOrderFrame<Node, Iter> {
    node: Node,
    iter: Iter,
}

pub fn dominators<G: ControlFlowGraph>(graph: G) -> Dominators<G::Node> {
    // compute the post order index (rank) for each node
    let mut post_order_rank = IndexVec::from_elem_n(0, graph.num_nodes());
    let mut parent: IndexVec<usize, Option<usize>> = IndexVec::from_elem_n(None, graph.num_nodes());

    let mut stack = vec![PreOrderFrame { node: 0, iter: graph.successors(graph.start_node()) }];
    let mut pre_order_to_real = Vec::with_capacity(graph.num_nodes());
    let mut real_to_pre_order: IndexVec<G::Node, Option<usize>> =
        IndexVec::from_elem_n(None, graph.num_nodes());
    pre_order_to_real.push(graph.start_node());
    real_to_pre_order[graph.start_node()] = Some(0);
    let mut idx = 1;
    let mut post_order_idx = 0;

    'recurse: while let Some(frame) = stack.last_mut() {
        while let Some(successor) = frame.iter.next() {
            if real_to_pre_order[successor].is_none() {
                real_to_pre_order[successor] = Some(idx);
                parent[idx] = Some(frame.node);
                pre_order_to_real.push(successor);

                stack.push(PreOrderFrame { node: idx, iter: graph.successors(successor) });
                idx += 1;
                continue 'recurse;
            }
        }
        post_order_rank[pre_order_to_real[frame.node]] = post_order_idx;
        post_order_idx += 1;

        stack.pop();
    }

    let mut idom = IndexVec::from_elem_n(0, pre_order_to_real.len());
    let mut semi = IndexVec::from_fn_n(std::convert::identity, pre_order_to_real.len());
    let mut label = semi.clone();
    let mut bucket = IndexVec::from_elem_n(vec![], pre_order_to_real.len());
    let mut lastlinked = None;

    for w in (1..pre_order_to_real.len()).rev() {
        // Optimization: process buckets just once, at the start of the
        // iteration. Do not explicitly empty the bucket (even though it will
        // not be used again), to save some instructions.
        let z = parent[w].unwrap();
        for &v in bucket[z].iter() {
            let y = eval(&mut parent, lastlinked, &semi, &mut label, v);
            idom[v] = if semi[y] < z { y } else { z };
        }

        semi[w] = w;
        for v in graph.predecessors(pre_order_to_real[w]) {
            let v = real_to_pre_order[v].unwrap();
            let x = eval(&mut parent, lastlinked, &semi, &mut label, v);
            semi[w] = std::cmp::min(semi[w], semi[x]);
        }
        // semi[w] is now semidominator(w).

        // Optimization: Do not insert into buckets if parent[w] = semi[w], as
        // we then immediately know the idom.
        if parent[w].unwrap() != semi[w] {
            bucket[semi[w]].push(w);
        } else {
            idom[w] = parent[w].unwrap();
        }

        // Optimization: We share the parent array between processed and not
        // processed elements; lastlinked represents the divider.
        lastlinked = Some(w);
    }
    for w in 1..pre_order_to_real.len() {
        if idom[w] != semi[w] {
            idom[w] = idom[idom[w]];
        }
    }

    let mut immediate_dominators = IndexVec::from_elem_n(None, graph.num_nodes());
    for (idx, node) in pre_order_to_real.iter().enumerate() {
        immediate_dominators[*node] = Some(pre_order_to_real[idom[idx]]);
    }

    Dominators { post_order_rank, immediate_dominators }
}

fn eval<N: Idx>(
    ancestor: &mut IndexVec<N, Option<N>>,
    lastlinked: Option<N>,
    semi: &IndexVec<N, N>,
    label: &mut IndexVec<N, N>,
    node: N,
) -> N {
    if is_processed(node, lastlinked) {
        compress(ancestor, lastlinked, semi, label, node);
        label[node]
    } else {
        node
    }
}

fn is_processed<N: Idx>(v: N, lastlinked: Option<N>) -> bool {
    if let Some(ll) = lastlinked { v >= ll } else { false }
}

fn compress<N: Idx>(
    ancestor: &mut IndexVec<N, Option<N>>,
    lastlinked: Option<N>,
    semi: &IndexVec<N, N>,
    label: &mut IndexVec<N, N>,
    v: N,
) {
    assert!(is_processed(v, lastlinked));
    let u = ancestor[v].unwrap();
    if is_processed(u, lastlinked) {
        compress(ancestor, lastlinked, semi, label, u);
        if semi[label[u]] < semi[label[v]] {
            label[v] = label[u];
        }
        ancestor[v] = ancestor[u];
    }
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
