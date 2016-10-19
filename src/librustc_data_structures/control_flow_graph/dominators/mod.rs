// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Algorithm citation:
//! A Simple, Fast Dominance Algorithm.
//! Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
//! Rice Computer Science TS-06-33870
//! https://www.cs.rice.edu/~keith/EMBED/dom.pdf

use super::ControlFlowGraph;
use super::iterate::reverse_post_order;
use super::super::indexed_vec::{IndexVec, Idx};

use std::fmt;

#[cfg(test)]
mod test;

pub fn dominators<G: ControlFlowGraph>(graph: &G) -> Dominators<G::Node> {
    let start_node = graph.start_node();
    let rpo = reverse_post_order(graph, start_node);
    dominators_given_rpo(graph, &rpo)
}

pub fn dominators_given_rpo<G: ControlFlowGraph>(graph: &G,
                                                 rpo: &[G::Node])
                                                 -> Dominators<G::Node> {
    let start_node = graph.start_node();
    assert_eq!(rpo[0], start_node);

    // compute the post order index (rank) for each node
    let mut post_order_rank: IndexVec<G::Node, usize> = IndexVec::from_elem_n(usize::default(),
                                                                              graph.num_nodes());
    for (index, node) in rpo.iter().rev().cloned().enumerate() {
        post_order_rank[node] = index;
    }

    let mut immediate_dominators: IndexVec<G::Node, Option<G::Node>> =
        IndexVec::from_elem_n(Option::default(), graph.num_nodes());
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
                    new_idom = intersect_opt(&post_order_rank,
                                             &immediate_dominators,
                                             new_idom,
                                             Some(pred));
                }
            }

            if new_idom != immediate_dominators[node] {
                immediate_dominators[node] = new_idom;
                changed = true;
            }
        }
    }

    Dominators {
        post_order_rank: post_order_rank,
        immediate_dominators: immediate_dominators,
    }
}

fn intersect_opt<Node: Idx>(post_order_rank: &IndexVec<Node, usize>,
                            immediate_dominators: &IndexVec<Node, Option<Node>>,
                            node1: Option<Node>,
                            node2: Option<Node>)
                            -> Option<Node> {
    match (node1, node2) {
        (None, None) => None,
        (Some(n), None) | (None, Some(n)) => Some(n),
        (Some(n1), Some(n2)) => Some(intersect(post_order_rank, immediate_dominators, n1, n2)),
    }
}

fn intersect<Node: Idx>(post_order_rank: &IndexVec<Node, usize>,
                        immediate_dominators: &IndexVec<Node, Option<Node>>,
                        mut node1: Node,
                        mut node2: Node)
                        -> Node {
    while node1 != node2 {
        while post_order_rank[node1] < post_order_rank[node2] {
            node1 = immediate_dominators[node1].unwrap();
        }

        while post_order_rank[node2] < post_order_rank[node1] {
            node2 = immediate_dominators[node2].unwrap();
        }
    }
    return node1;
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

    pub fn dominators(&self, node: Node) -> Iter<Node> {
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

    pub fn mutual_dominator_node(&self, node1: Node, node2: Node) -> Node {
        assert!(self.is_reachable(node1),
                "node {:?} is not reachable",
                node1);
        assert!(self.is_reachable(node2),
                "node {:?} is not reachable",
                node2);
        intersect::<Node>(&self.post_order_rank,
                          &self.immediate_dominators,
                          node1,
                          node2)
    }

    pub fn mutual_dominator<I>(&self, iter: I) -> Option<Node>
        where I: IntoIterator<Item = Node>
    {
        let mut iter = iter.into_iter();
        iter.next()
            .map(|dom| iter.fold(dom, |dom, node| self.mutual_dominator_node(dom, node)))
    }

    pub fn all_immediate_dominators(&self) -> &IndexVec<Node, Option<Node>> {
        &self.immediate_dominators
    }

    pub fn dominator_tree(&self) -> DominatorTree<Node> {
        let elem: Vec<Node> = Vec::new();
        let mut children: IndexVec<Node, Vec<Node>> =
            IndexVec::from_elem_n(elem, self.immediate_dominators.len());
        let mut root = None;
        for (index, immed_dom) in self.immediate_dominators.iter().enumerate() {
            let node = Node::new(index);
            match *immed_dom {
                None => {
                    // node not reachable
                }
                Some(immed_dom) => {
                    if node == immed_dom {
                        root = Some(node);
                    } else {
                        children[immed_dom].push(node);
                    }
                }
            }
        }
        DominatorTree {
            root: root.unwrap(),
            children: children,
        }
    }
}

pub struct Iter<'dom, Node: Idx + 'dom> {
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

pub struct DominatorTree<N: Idx> {
    root: N,
    children: IndexVec<N, Vec<N>>,
}

impl<Node: Idx> DominatorTree<Node> {
    pub fn root(&self) -> Node {
        self.root
    }

    pub fn children(&self, node: Node) -> &[Node] {
        &self.children[node]
    }

    pub fn iter_children_of(&self, node: Node) -> IterChildrenOf<Node> {
        IterChildrenOf {
            tree: self,
            stack: vec![node],
        }
    }
}

pub struct IterChildrenOf<'iter, Node: Idx + 'iter> {
    tree: &'iter DominatorTree<Node>,
    stack: Vec<Node>,
}

impl<'iter, Node: Idx> Iterator for IterChildrenOf<'iter, Node> {
    type Item = Node;

    fn next(&mut self) -> Option<Node> {
        if let Some(node) = self.stack.pop() {
            self.stack.extend(self.tree.children(node));
            Some(node)
        } else {
            None
        }
    }
}

impl<Node: Idx> fmt::Debug for DominatorTree<Node> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(&DominatorTreeNode {
                            tree: self,
                            node: self.root,
                        },
                        fmt)
    }
}

struct DominatorTreeNode<'tree, Node: Idx> {
    tree: &'tree DominatorTree<Node>,
    node: Node,
}

impl<'tree, Node: Idx> fmt::Debug for DominatorTreeNode<'tree, Node> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let subtrees: Vec<_> = self.tree
            .children(self.node)
            .iter()
            .map(|&child| {
                DominatorTreeNode {
                    tree: self.tree,
                    node: child,
                }
            })
            .collect();
        fmt.debug_tuple("")
            .field(&self.node)
            .field(&subtrees)
            .finish()
    }
}
