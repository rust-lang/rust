// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Module that constructs a control-flow graph representing an item.
Uses `Graph` as the underlying representation.

*/

use middle::graph;
use middle::ty;
use syntax::ast;
use util::nodemap::NodeMap;

mod construct;
pub mod graphviz;

pub struct CFG {
    pub exit_map: NodeMap<CFGIndex>,
    pub graph: CFGGraph,
    pub entry: CFGIndex,
    pub exit: CFGIndex,
}

pub struct CFGNodeData {
    pub id: ast::NodeId
}

pub struct CFGEdgeData {
    pub exiting_scopes: Vec<ast::NodeId>
}

pub type CFGIndex = graph::NodeIndex;

pub type CFGGraph = graph::Graph<CFGNodeData, CFGEdgeData>;

pub type CFGNode = graph::Node<CFGNodeData>;

pub type CFGEdge = graph::Edge<CFGEdgeData>;

impl CFG {
    pub fn new(tcx: &ty::ctxt,
               blk: &ast::Block) -> CFG {
        construct::construct(tcx, blk)
    }
}
