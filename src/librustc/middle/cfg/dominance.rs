// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implements the Rice algorithm to determine dominators and postdominators
//! for each node in a control flow graph.

use middle::cfg::{CFG, CFGIndex};
use middle::graph::{Direction, Incoming, NodeIndex, Outgoing};

use collections::bitv::BitvSet;

pub struct Dominators {
    dominators: Vec<CFGIndex>,
    postdominators: Vec<CFGIndex>,
}

impl Dominators {
    pub fn new(cfg: &CFG) -> Dominators {
        Dominators {
            dominators: compute_dominators(cfg, Outgoing),
            postdominators: compute_dominators(cfg, Incoming),
        }
    }

    /// Kind of a kludge. Creates a "blank" `Dominators` instance that you
    /// are not allowed to touch.
    pub fn blank() -> Dominators {
        Dominators {
            dominators: Vec::new(),
            postdominators: Vec::new(),
        }
    }

    pub fn print(&self) {
        println!("Dominators:");
        for (i, dom) in self.dominators.iter().enumerate() {
            println!("{}: {}", i, dom);
        }
        println!("Postdominators:");
        for (i, dom) in self.postdominators.iter().enumerate() {
            println!("{}: {}", i, dom);
        }
    }
}

fn compute_reverse_postorder(cfg: &CFG, direction: Direction)
                             -> Vec<CFGIndex> {
    fn go(cfg: &CFG,
          node: CFGIndex,
          direction: Direction,
          accumulator: &mut Vec<CFGIndex>,
          breadcrumbs: &mut BitvSet) {
        if breadcrumbs.contains(&node.node_id()) {
            return
        }
        breadcrumbs.insert(node.node_id());

        cfg.graph.each_adjacent_edge(node, direction, |_, edge| {
            let next = if direction == Outgoing {
                edge.target
            } else {
                edge.source
            };
            go(cfg, next, direction, accumulator, breadcrumbs);
            true
        });

        accumulator.push(node);
    }

    let mut result = Vec::new();
    let origin = if direction == Outgoing {
        cfg.entry
    } else {
        cfg.exit
    };
    go(cfg, origin, direction, &mut result, &mut BitvSet::new());
    result.reverse();
    result
}

fn compute_inverse_mapping(reverse_postorder_index_to_node_id: &[CFGIndex])
                           -> Vec<uint> {
    let mut node_id_to_reverse_postorder_index =
        Vec::from_elem(reverse_postorder_index_to_node_id.len(), -1);
    for (reverse_postorder_index, node_id) in
            reverse_postorder_index_to_node_id.iter().enumerate() {
        node_id_to_reverse_postorder_index.grow_set(
            node_id.node_id(),
            &-1,
            reverse_postorder_index)
    }
    node_id_to_reverse_postorder_index
}

fn compute_dominators(cfg: &CFG, direction: Direction) -> Vec<CFGIndex> {
    // Compute the reverse postorder index.
    let reverse_postorder_index_to_node_id =
        compute_reverse_postorder(cfg, direction);
    let node_id_to_reverse_postorder_index = compute_inverse_mapping(
        reverse_postorder_index_to_node_id.as_slice());

    // Initialize.
    let mut dominators =
        Vec::from_elem(reverse_postorder_index_to_node_id.len(), -1);
    *dominators.get_mut(0) = 0;

    loop {
        let mut changed = false;
        for node in range(1, reverse_postorder_index_to_node_id.len()) {
            let mut new_idom = -1;
            cfg.graph
               .each_adjacent_edge(reverse_postorder_index_to_node_id[node],
                                   direction.reverse(),
                                   |_, predecessor_edge| {
                let predecessor = if direction == Outgoing {
                    predecessor_edge.source
                } else {
                    predecessor_edge.target
                };
                let predecessor = predecessor.node_id();
                let predecessor =
                    if predecessor <
                            node_id_to_reverse_postorder_index.len() {
                       node_id_to_reverse_postorder_index[predecessor]
                    } else {
                        -1
                    };
                // Check to make sure the node isn't dead!
                if predecessor != -1 && predecessor < dominators.len() &&
                            dominators[predecessor] != -1 {
                    new_idom = if new_idom == -1 {
                        predecessor
                    } else {
                        intersect(dominators.as_slice(),
                                  predecessor,
                                  new_idom)
                    }
                }
                true
            });

            if new_idom != dominators[node] {
                *dominators.get_mut(node) = new_idom;
                changed = true
            }
        }

        if !changed {
            break
        }
    }

    let mut result = Vec::from_elem(cfg.graph.next_node_index().node_id(),
                                    NodeIndex(-1));
    for (node, dominator) in dominators.iter().enumerate() {
        if *dominator == -1 {
            continue
        }
        let node_id = reverse_postorder_index_to_node_id[node];
        let dominator_id = reverse_postorder_index_to_node_id[*dominator];
        *result.get_mut(node_id.node_id()) = dominator_id
    }
    result
}

fn intersect(dominators: &[uint], mut finger1: uint, mut finger2: uint)
             -> uint {
    while finger1 != finger2 {
        while finger1 > finger2 {
            finger1 = dominators[finger1]
        }
        while finger2 > finger1 {
            finger2 = dominators[finger2]
        }
    }
    finger1
}

