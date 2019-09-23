//! Module that constructs a control-flow graph representing an item.
//! Uses `Graph` as the underlying representation.

use rustc_data_structures::graph::implementation as graph;
use rustc::ty::TyCtxt;
use rustc::hir;
use rustc::hir::def_id::DefId;

mod construct;
pub mod graphviz;

pub struct CFG {
    owner_def_id: DefId,
    pub(crate) graph: CFGGraph,
    pub(crate) entry: CFGIndex,
    exit: CFGIndex,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CFGNodeData {
    AST(hir::ItemLocalId),
    Entry,
    Exit,
    Dummy,
    Unreachable,
}

impl CFGNodeData {
    pub(crate) fn id(&self) -> hir::ItemLocalId {
        if let CFGNodeData::AST(id) = *self {
            id
        } else {
            hir::DUMMY_ITEM_LOCAL_ID
        }
    }
}

#[derive(Debug)]
pub struct CFGEdgeData {
    pub(crate) exiting_scopes: Vec<hir::ItemLocalId>
}

pub(crate) type CFGIndex = graph::NodeIndex;

pub(crate) type CFGGraph = graph::Graph<CFGNodeData, CFGEdgeData>;

pub(crate) type CFGNode = graph::Node<CFGNodeData>;

pub(crate) type CFGEdge = graph::Edge<CFGEdgeData>;

impl CFG {
    pub fn new(tcx: TyCtxt<'_>, body: &hir::Body) -> CFG {
        construct::construct(tcx, body)
    }
}
