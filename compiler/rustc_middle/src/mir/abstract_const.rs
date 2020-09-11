//! A subset of a mir body used for const evaluatability checking.
use crate::mir;
use crate::ty;

rustc_index::newtype_index! {
    /// An index into an `AbstractConst`.
    pub struct NodeId {
        derive [HashStable]
        DEBUG_FORMAT = "n{}",
    }
}

/// A node of an `AbstractConst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum Node<'tcx> {
    Leaf(&'tcx ty::Const<'tcx>),
    Binop(mir::BinOp, NodeId, NodeId),
    UnaryOp(mir::UnOp, NodeId),
    FunctionCall(NodeId, &'tcx [NodeId]),
}
