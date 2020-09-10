//! A subset of a mir body used for const evaluatability checking.
use crate::mir;
use crate::ty;

/// An index into an `AbstractConst`.
pub type NodeId = usize;

/// A node of an `AbstractConst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, HashStable)]
pub enum Node<'tcx> {
    Leaf(&'tcx ty::Const<'tcx>),
    Binop(mir::BinOp, NodeId, NodeId),
    UnaryOp(mir::UnOp, NodeId),
    FunctionCall(NodeId, &'tcx [NodeId]),
}
