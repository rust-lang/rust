// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Module that constructs a control-flow graph representing an item.
//! Uses `Graph` as the underlying representation.

use rustc_data_structures::graph;
use ty::TyCtxt;
use syntax::ast;
use hir;

mod construct;
pub mod graphviz;

pub struct CFG {
    pub graph: CFGGraph,
    pub entry: CFGIndex,
    pub exit: CFGIndex,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CFGNodeData {
    AST(ast::NodeId),
    Entry,
    Exit,
    Dummy,
    Unreachable,
}

impl CFGNodeData {
    pub fn id(&self) -> ast::NodeId {
        if let CFGNodeData::AST(id) = *self {
            id
        } else {
            ast::DUMMY_NODE_ID
        }
    }
}

#[derive(Debug)]
pub struct CFGEdgeData {
    pub exiting_scopes: Vec<ast::NodeId>
}

pub type CFGIndex = graph::NodeIndex;

pub type CFGGraph = graph::Graph<CFGNodeData, CFGEdgeData>;

pub type CFGNode = graph::Node<CFGNodeData>;

pub type CFGEdge = graph::Edge<CFGEdgeData>;

impl CFG {
    pub fn new<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         body: &hir::Expr) -> CFG {
        construct::construct(tcx, body)
    }

    pub fn node_is_reachable(&self, id: ast::NodeId) -> bool {
        self.graph.depth_traverse(self.entry, graph::OUTGOING)
                  .any(|idx| self.graph.node_data(idx).id() == id)
    }
}
