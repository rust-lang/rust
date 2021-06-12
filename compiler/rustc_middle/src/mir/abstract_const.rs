//! A subset of a mir body used for const evaluatability checking.
use crate::mir::{self, CastKind};
use crate::ty::{self, Ty};

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
    Cast(CastKind, NodeId, Ty<'tcx>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum NotConstEvaluatable {
    Error(rustc_errors::ErrorReported),
    MentionsInfer,
    MentionsParam,
}

impl From<rustc_errors::ErrorReported> for NotConstEvaluatable {
    fn from(e: rustc_errors::ErrorReported) -> NotConstEvaluatable {
        NotConstEvaluatable::Error(e)
    }
}

TrivialTypeFoldableAndLiftImpls! {
    NotConstEvaluatable,
}
