// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! MIR-based callgraph.
//!
//! This only considers direct calls

use rustc::hir::def_id::DefId;
use rustc_data_structures::graph;

use rustc::mir::repr::*;
use rustc::mir::visit::*;
use rustc::mir::mir_map::MirMap;

use rustc::ty;

use rustc::util::nodemap::DefIdMap;

pub struct CallGraph {
    node_map: DefIdMap<graph::NodeIndex>,
    graph: graph::Graph<DefId, ()>
}

impl CallGraph {
    pub fn build<'tcx>(map: &MirMap<'tcx>) -> CallGraph {
        let def_ids = map.map.keys();

        let mut callgraph = CallGraph {
            node_map: DefIdMap(),
            graph: graph::Graph::new()
        };

        for def_id in def_ids {
            let idx = callgraph.add_node(def_id);

            let mut call_visitor = CallVisitor {
                caller: idx,
                graph: &mut callgraph
            };

            let mir = map.map.get(&def_id).unwrap();
            call_visitor.visit_mir(&mir);
        }

        callgraph
    }

    pub fn scc_iter<'g>(&'g self) -> SCCIterator<'g> {
        SCCIterator::new(&self.graph)
    }

    pub fn def_id(&self, node: graph::NodeIndex) -> DefId {
        *self.graph.node_data(node)
    }

    fn add_node(&mut self, id: DefId) -> graph::NodeIndex {
        let graph = &mut self.graph;
        *self.node_map.entry(id).or_insert_with(|| {
            graph.add_node(id)
        })
    }
}

struct CallVisitor<'a> {
    caller: graph::NodeIndex,
    graph: &'a mut CallGraph
}

impl<'a, 'tcx> Visitor<'tcx> for CallVisitor<'a> {
    fn visit_terminator_kind(&mut self, _block: BasicBlock,
                             kind: &TerminatorKind<'tcx>, _loc: Location) {
        if let TerminatorKind::Call {
            func: Operand::Constant(ref f)
            , .. } = *kind {
            if let ty::TyFnDef(def_id, _, _) = f.ty.sty {
                let callee = self.graph.add_node(def_id);
                self.graph.graph.add_edge(self.caller, callee, ());
            }
        }
    }
}

struct StackElement<'g> {
    node: graph::NodeIndex,
    lowlink: usize,
    children: graph::AdjacentTargets<'g, DefId, ()>
}

pub struct SCCIterator<'g> {
    graph: &'g graph::Graph<DefId, ()>,
    index: usize,
    node_indices: Vec<Option<usize>>,
    scc_stack: Vec<graph::NodeIndex>,
    current_scc: Vec<graph::NodeIndex>,
    visit_stack: Vec<StackElement<'g>>,
}

impl<'g> SCCIterator<'g> {
    pub fn new(graph: &'g graph::Graph<DefId, ()>) -> SCCIterator<'g> {
        if graph.len_nodes() == 0 {
            return SCCIterator {
                graph: graph,
                index: 0,
                node_indices: Vec::new(),
                scc_stack: Vec::new(),
                current_scc: Vec::new(),
                visit_stack: Vec::new()
            };
        }

        let first = graph::NodeIndex(0);

        SCCIterator::with_entry(graph, first)
    }

    pub fn with_entry(graph: &'g graph::Graph<DefId, ()>,
                      entry: graph::NodeIndex) -> SCCIterator<'g> {
        let mut iter = SCCIterator {
            graph: graph,
            index: 0,
            node_indices: Vec::with_capacity(graph.len_nodes()),
            scc_stack: Vec::new(),
            current_scc: Vec::new(),
            visit_stack: Vec::new()
        };

        iter.visit_one(entry);

        iter
    }

    fn get_next(&mut self) {
        self.current_scc.clear();

        while !self.visit_stack.is_empty() {
            self.visit_children();

            let node = self.visit_stack.pop().unwrap();

            if let Some(last) = self.visit_stack.last_mut() {
                if last.lowlink > node.lowlink {
                    last.lowlink = node.lowlink;
                }
            }

            debug!("TarjanSCC: Popped node {:?} : lowlink = {:?}; index = {:?}",
                   node.node, node.lowlink, self.node_index(node.node).unwrap());

            if node.lowlink != self.node_index(node.node).unwrap() {
                continue;
            }

            loop {
                let n = self.scc_stack.pop().unwrap();
                self.current_scc.push(n);
                self.set_node_index(n, !0);
                if n == node.node { return; }
            }
        }
    }

    fn visit_one(&mut self, node: graph::NodeIndex) {
        self.index += 1;
        let idx =  self.index;
        self.set_node_index(node, idx);
        self.scc_stack.push(node);
        self.visit_stack.push(StackElement {
            node: node,
            lowlink: self.index,
            children: self.graph.successor_nodes(node)
        });
        debug!("TarjanSCC: Node {:?} : index = {:?}", node, idx);
    }

    fn visit_children(&mut self) {
        while let Some(child) = self.visit_stack.last_mut().unwrap().children.next() {
            if let Some(child_num) = self.node_index(child) {
                let cur = self.visit_stack.last_mut().unwrap();
                if cur.lowlink > child_num {
                    cur.lowlink = child_num;
                }
            } else {
                self.visit_one(child);
            }
        }
    }

    fn node_index(&self, node: graph::NodeIndex) -> Option<usize> {
        self.node_indices.get(node.node_id()).and_then(|&idx| idx)
    }

    fn set_node_index(&mut self, node: graph::NodeIndex, idx: usize) {
        let i = node.node_id();
        if i >= self.node_indices.len() {
            self.node_indices.resize(i + 1, None);
        }
        self.node_indices[i] = Some(idx);
    }
}

impl<'g> Iterator for SCCIterator<'g> {
    type Item = Vec<graph::NodeIndex>;

    fn next(&mut self) -> Option<Vec<graph::NodeIndex>> {
        self.get_next();

        if self.current_scc.is_empty() {
            // Try a new root for the next SCC, if the node_indices
            // map is doesn't contain all nodes, use the smallest one
            // with no entry, otherwise find the first empty node.
            //
            // FIXME: This should probably use a set of precomputed
            // roots instead
            if self.node_indices.len() < self.graph.len_nodes() {
                let idx = graph::NodeIndex(self.node_indices.len());
                self.visit_one(idx);
            } else {
                for idx in 0..self.node_indices.len() {
                    if self.node_indices[idx].is_none() {
                        let idx = graph::NodeIndex(idx);
                        self.visit_one(idx);
                        break;
                    }
                }
            }
            self.get_next();
        }

        if self.current_scc.is_empty() {
            None
        } else {
            Some(self.current_scc.clone())
        }
    }
}
