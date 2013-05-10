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
use middle::typeck;
use std::hashmap::HashMap;
use syntax::ast;
use syntax::opt_vec::OptVec;

mod construct;

pub struct CFG {
    exit_map: HashMap<ast::node_id, CFGIndex>,
    graph: CFGGraph,
    entry: CFGIndex,
    exit: CFGIndex,
}

pub struct CFGNodeData {
    id: ast::node_id
}

pub struct CFGEdgeData {
    exiting_scopes: OptVec<ast::node_id>
}

pub type CFGIndex = graph::NodeIndex;

pub type CFGGraph = graph::Graph<CFGNodeData, CFGEdgeData>;

pub type CFGNode = graph::Node<CFGNodeData>;

pub type CFGEdge = graph::Edge<CFGEdgeData>;

pub struct CFGIndices {
    entry: CFGIndex,
    exit: CFGIndex,
}

impl CFG {
    pub fn new(tcx: ty::ctxt,
               method_map: typeck::method_map,
               blk: &ast::blk) -> CFG {
        construct::construct(tcx, method_map, blk)
    }
}