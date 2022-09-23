//! A subset of a mir body used for const evaluatability checking.
use crate::mir;
use crate::ty::visit::TypeVisitable;
use crate::ty::{self, DelaySpanBugEmitted, EarlyBinder, SubstsRef, Ty, TyCtxt};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;
use std::cmp;
use std::ops::ControlFlow;

rustc_index::newtype_index! {
    /// An index into an `AbstractConst`.
    pub struct NodeId {
        derive [HashStable]
        DEBUG_FORMAT = "n{}",
    }
}

/// A tree representing an anonymous constant.
///
/// This is only able to represent a subset of `MIR`,
/// and should not leak any information about desugarings.
#[derive(Debug, Clone, Copy)]
pub struct AbstractConst<'tcx> {
    // FIXME: Consider adding something like `IndexSlice`
    // and use this here.
    inner: &'tcx [Node<'tcx>],
    substs: SubstsRef<'tcx>,
}

impl<'tcx> AbstractConst<'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        uv: ty::UnevaluatedConst<'tcx>,
    ) -> Result<Option<AbstractConst<'tcx>>, ErrorGuaranteed> {
        let inner = tcx.thir_abstract_const_opt_const_arg(uv.def)?;
        debug!("AbstractConst::new({:?}) = {:?}", uv, inner);
        Ok(inner.map(|inner| AbstractConst { inner, substs: tcx.erase_regions(uv.substs) }))
    }

    pub fn from_const(
        tcx: TyCtxt<'tcx>,
        ct: ty::Const<'tcx>,
    ) -> Result<Option<AbstractConst<'tcx>>, ErrorGuaranteed> {
        match ct.kind() {
            ty::ConstKind::Unevaluated(uv) => AbstractConst::new(tcx, uv),
            ty::ConstKind::Error(DelaySpanBugEmitted { reported, .. }) => Err(reported),
            _ => Ok(None),
        }
    }

    #[inline]
    pub fn subtree(self, node: NodeId) -> AbstractConst<'tcx> {
        AbstractConst { inner: &self.inner[..=node.index()], substs: self.substs }
    }

    #[inline]
    pub fn root(self, tcx: TyCtxt<'tcx>) -> Node<'tcx> {
        let node = self.inner.last().copied().unwrap();
        match node {
            Node::Leaf(leaf) => Node::Leaf(EarlyBinder(leaf).subst(tcx, self.substs)),
            Node::Cast(kind, operand, ty) => {
                Node::Cast(kind, operand, EarlyBinder(ty).subst(tcx, self.substs))
            }
            // Don't perform substitution on the following as they can't directly contain generic params
            Node::Binop(_, _, _) | Node::UnaryOp(_, _) | Node::FunctionCall(_, _) => node,
        }
    }

    pub fn unify_failure_kind(self, tcx: TyCtxt<'tcx>) -> FailureKind {
        let mut failure_kind = FailureKind::Concrete;
        walk_abstract_const::<!, _>(tcx, self, |node| {
            match node.root(tcx) {
                Node::Leaf(leaf) => {
                    if leaf.has_infer_types_or_consts() {
                        failure_kind = FailureKind::MentionsInfer;
                    } else if leaf.has_param_types_or_consts() {
                        failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                    }
                }
                Node::Cast(_, _, ty) => {
                    if ty.has_infer_types_or_consts() {
                        failure_kind = FailureKind::MentionsInfer;
                    } else if ty.has_param_types_or_consts() {
                        failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                    }
                }
                Node::Binop(_, _, _) | Node::UnaryOp(_, _) | Node::FunctionCall(_, _) => {}
            }
            ControlFlow::CONTINUE
        });
        failure_kind
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum CastKind {
    /// thir::ExprKind::As
    As,
    /// thir::ExprKind::Use
    Use,
}

/// A node of an `AbstractConst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum Node<'tcx> {
    Leaf(ty::Const<'tcx>),
    Binop(mir::BinOp, NodeId, NodeId),
    UnaryOp(mir::UnOp, NodeId),
    FunctionCall(NodeId, &'tcx [NodeId]),
    Cast(CastKind, NodeId, Ty<'tcx>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum NotConstEvaluatable {
    Error(ErrorGuaranteed),
    MentionsInfer,
    MentionsParam,
}

impl From<ErrorGuaranteed> for NotConstEvaluatable {
    fn from(e: ErrorGuaranteed) -> NotConstEvaluatable {
        NotConstEvaluatable::Error(e)
    }
}

TrivialTypeTraversalAndLiftImpls! {
    NotConstEvaluatable,
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline]
    pub fn thir_abstract_const_opt_const_arg(
        self,
        def: ty::WithOptConstParam<DefId>,
    ) -> Result<Option<&'tcx [Node<'tcx>]>, ErrorGuaranteed> {
        if let Some((did, param_did)) = def.as_const_arg() {
            self.thir_abstract_const_of_const_arg((did, param_did))
        } else {
            self.thir_abstract_const(def.did)
        }
    }
}

#[instrument(skip(tcx, f), level = "debug")]
pub fn walk_abstract_const<'tcx, R, F>(
    tcx: TyCtxt<'tcx>,
    ct: AbstractConst<'tcx>,
    mut f: F,
) -> ControlFlow<R>
where
    F: FnMut(AbstractConst<'tcx>) -> ControlFlow<R>,
{
    #[instrument(skip(tcx, f), level = "debug")]
    fn recurse<'tcx, R>(
        tcx: TyCtxt<'tcx>,
        ct: AbstractConst<'tcx>,
        f: &mut dyn FnMut(AbstractConst<'tcx>) -> ControlFlow<R>,
    ) -> ControlFlow<R> {
        f(ct)?;
        let root = ct.root(tcx);
        debug!(?root);
        match root {
            Node::Leaf(_) => ControlFlow::CONTINUE,
            Node::Binop(_, l, r) => {
                recurse(tcx, ct.subtree(l), f)?;
                recurse(tcx, ct.subtree(r), f)
            }
            Node::UnaryOp(_, v) => recurse(tcx, ct.subtree(v), f),
            Node::FunctionCall(func, args) => {
                recurse(tcx, ct.subtree(func), f)?;
                args.iter().try_for_each(|&arg| recurse(tcx, ct.subtree(arg), f))
            }
            Node::Cast(_, operand, _) => recurse(tcx, ct.subtree(operand), f),
        }
    }

    recurse(tcx, ct, &mut f)
}

// We were unable to unify the abstract constant with
// a constant found in the caller bounds, there are
// now three possible cases here.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FailureKind {
    /// The abstract const still references an inference
    /// variable, in this case we return `TooGeneric`.
    MentionsInfer,
    /// The abstract const references a generic parameter,
    /// this means that we emit an error here.
    MentionsParam,
    /// The substs are concrete enough that we can simply
    /// try and evaluate the given constant.
    Concrete,
}
