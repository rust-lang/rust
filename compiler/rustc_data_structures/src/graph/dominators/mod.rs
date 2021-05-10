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

struct PreOrderFrame<Iter> {
    pre_order_idx: PreorderIndex,
    iter: Iter,
}

rustc_index::newtype_index! {
    struct PreorderIndex { .. }
}

pub fn dominators<G: ControlFlowGraph>(graph: G) -> Dominators<G::Node> {
    // compute the post order index (rank) for each node
    let mut post_order_rank = IndexVec::from_elem_n(0, graph.num_nodes());

    // We allocate capacity for the full set of nodes, because most of the time
    // most of the nodes *are* reachable.
    let mut parent: IndexVec<PreorderIndex, PreorderIndex> =
        IndexVec::with_capacity(graph.num_nodes());

    let mut stack = vec![PreOrderFrame {
        pre_order_idx: PreorderIndex::new(0),
        iter: graph.successors(graph.start_node()),
    }];
    let mut pre_order_to_real: IndexVec<PreorderIndex, G::Node> =
        IndexVec::with_capacity(graph.num_nodes());
    let mut real_to_pre_order: IndexVec<G::Node, Option<PreorderIndex>> =
        IndexVec::from_elem_n(None, graph.num_nodes());
    pre_order_to_real.push(graph.start_node());
    parent.push(PreorderIndex::new(0)); // the parent of the root node is the root for now.
    real_to_pre_order[graph.start_node()] = Some(PreorderIndex::new(0));
    let mut post_order_idx = 0;

    'recurse: while let Some(frame) = stack.last_mut() {
        while let Some(successor) = frame.iter.next() {
            if real_to_pre_order[successor].is_none() {
                let pre_order_idx = pre_order_to_real.push(successor);
                real_to_pre_order[successor] = Some(pre_order_idx);
                parent.push(frame.pre_order_idx);
                stack.push(PreOrderFrame { pre_order_idx, iter: graph.successors(successor) });

                continue 'recurse;
            }
        }
        post_order_rank[pre_order_to_real[frame.pre_order_idx]] = post_order_idx;
        post_order_idx += 1;

        stack.pop();
    }

    let reachable_vertices = pre_order_to_real.len();

    let mut idom = IndexVec::from_elem_n(PreorderIndex::new(0), reachable_vertices);
    let mut semi = IndexVec::from_fn_n(std::convert::identity, reachable_vertices);
    let mut label = semi.clone();
    let mut bucket = IndexVec::from_elem_n(vec![], reachable_vertices);
    let mut lastlinked = None;

    for w in (PreorderIndex::new(1)..PreorderIndex::new(reachable_vertices)).rev() {
        // Optimization: process buckets just once, at the start of the
        // iteration. Do not explicitly empty the bucket (even though it will
        // not be used again), to save some instructions.
        let z = parent[w];
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
        if parent[w] != semi[w] {
            bucket[semi[w]].push(w);
        } else {
            idom[w] = parent[w];
        }

        // Optimization: We share the parent array between processed and not
        // processed elements; lastlinked represents the divider.
        lastlinked = Some(w);
    }
    for w in PreorderIndex::new(1)..PreorderIndex::new(reachable_vertices) {
        if idom[w] != semi[w] {
            idom[w] = idom[idom[w]];
        }
    }

    let mut immediate_dominators = IndexVec::from_elem_n(None, graph.num_nodes());
    for (idx, node) in pre_order_to_real.iter_enumerated() {
        immediate_dominators[*node] = Some(pre_order_to_real[idom[idx]]);
    }

    Dominators { post_order_rank, immediate_dominators }
}

#[inline]
fn eval(
    ancestor: &mut IndexVec<PreorderIndex, PreorderIndex>,
    lastlinked: Option<PreorderIndex>,
    semi: &IndexVec<PreorderIndex, PreorderIndex>,
    label: &mut IndexVec<PreorderIndex, PreorderIndex>,
    node: PreorderIndex,
) -> PreorderIndex {
    if is_processed(node, lastlinked) {
        compress(ancestor, lastlinked, semi, label, node);
        label[node]
    } else {
        node
    }
}

#[inline]
fn is_processed(v: PreorderIndex, lastlinked: Option<PreorderIndex>) -> bool {
    if let Some(ll) = lastlinked { v >= ll } else { false }
}

#[inline]
fn compress(
    ancestor: &mut IndexVec<PreorderIndex, PreorderIndex>,
    lastlinked: Option<PreorderIndex>,
    semi: &IndexVec<PreorderIndex, PreorderIndex>,
    label: &mut IndexVec<PreorderIndex, PreorderIndex>,
    v: PreorderIndex,
) {
    assert!(is_processed(v, lastlinked));
    let u = ancestor[v];
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
