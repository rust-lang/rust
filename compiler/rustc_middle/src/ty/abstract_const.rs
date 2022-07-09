//! A subset of a mir body used for const evaluatability checking.
use crate::mir;
use crate::ty::{self, subst::Subst, DelaySpanBugEmitted, EarlyBinder, SubstsRef, Ty, TyCtxt};
use rustc_errors::ErrorGuaranteed;
use std::iter;
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
        uv: ty::Unevaluated<'tcx, ()>,
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
            ty::ConstKind::Unevaluated(uv) => AbstractConst::new(tcx, uv.shrink()),
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
        def: ty::WithOptConstParam<rustc_hir::def_id::DefId>,
    ) -> Result<Option<&'tcx [Node<'tcx>]>, ErrorGuaranteed> {
        if let Some((did, param_did)) = def.as_const_arg() {
            self.thir_abstract_const_of_const_arg((did, param_did))
        } else {
            self.thir_abstract_const(def.did)
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub fn try_unify_abstract_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    (a, b): (ty::Unevaluated<'tcx, ()>, ty::Unevaluated<'tcx, ()>),
    param_env: ty::ParamEnv<'tcx>,
) -> bool {
    (|| {
        if let Some(a) = AbstractConst::new(tcx, a)? {
            if let Some(b) = AbstractConst::new(tcx, b)? {
                let const_unify_ctxt = ConstUnifyCtxt { tcx, param_env };
                return Ok(const_unify_ctxt.try_unify(a, b));
            }
        }

        Ok(false)
    })()
    .unwrap_or_else(|_: ErrorGuaranteed| true)
    // FIXME(generic_const_exprs): We should instead have this
    // method return the resulting `ty::Const` and return `ConstKind::Error`
    // on `ErrorGuaranteed`.
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

pub struct ConstUnifyCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> ConstUnifyCtxt<'tcx> {
    // Substitutes generics repeatedly to allow AbstractConsts to unify where a
    // ConstKind::Unevaluated could be turned into an AbstractConst that would unify e.g.
    // Param(N) should unify with Param(T), substs: [Unevaluated("T2", [Unevaluated("T3", [Param(N)])])]
    #[inline]
    #[instrument(skip(self), level = "debug")]
    fn try_replace_substs_in_root(
        &self,
        mut abstr_const: AbstractConst<'tcx>,
    ) -> Option<AbstractConst<'tcx>> {
        while let Node::Leaf(ct) = abstr_const.root(self.tcx) {
            match AbstractConst::from_const(self.tcx, ct) {
                Ok(Some(act)) => abstr_const = act,
                Ok(None) => break,
                Err(_) => return None,
            }
        }

        Some(abstr_const)
    }

    /// Tries to unify two abstract constants using structural equality.
    #[instrument(skip(self), level = "debug")]
    pub fn try_unify(&self, a: AbstractConst<'tcx>, b: AbstractConst<'tcx>) -> bool {
        let a = if let Some(a) = self.try_replace_substs_in_root(a) {
            a
        } else {
            return true;
        };

        let b = if let Some(b) = self.try_replace_substs_in_root(b) {
            b
        } else {
            return true;
        };

        let a_root = a.root(self.tcx);
        let b_root = b.root(self.tcx);
        debug!(?a_root, ?b_root);

        match (a_root, b_root) {
            (Node::Leaf(a_ct), Node::Leaf(b_ct)) => {
                let a_ct = a_ct.eval(self.tcx, self.param_env);
                debug!("a_ct evaluated: {:?}", a_ct);
                let b_ct = b_ct.eval(self.tcx, self.param_env);
                debug!("b_ct evaluated: {:?}", b_ct);

                if a_ct.ty() != b_ct.ty() {
                    return false;
                }

                match (a_ct.kind(), b_ct.kind()) {
                    // We can just unify errors with everything to reduce the amount of
                    // emitted errors here.
                    (ty::ConstKind::Error(_), _) | (_, ty::ConstKind::Error(_)) => true,
                    (ty::ConstKind::Param(a_param), ty::ConstKind::Param(b_param)) => {
                        a_param == b_param
                    }
                    (ty::ConstKind::Value(a_val), ty::ConstKind::Value(b_val)) => a_val == b_val,
                    // If we have `fn a<const N: usize>() -> [u8; N + 1]` and `fn b<const M: usize>() -> [u8; 1 + M]`
                    // we do not want to use `assert_eq!(a(), b())` to infer that `N` and `M` have to be `1`. This
                    // means that we only allow inference variables if they are equal.
                    (ty::ConstKind::Infer(a_val), ty::ConstKind::Infer(b_val)) => a_val == b_val,
                    // We expand generic anonymous constants at the start of this function, so this
                    // branch should only be taking when dealing with associated constants, at
                    // which point directly comparing them seems like the desired behavior.
                    //
                    // FIXME(generic_const_exprs): This isn't actually the case.
                    // We also take this branch for concrete anonymous constants and
                    // expand generic anonymous constants with concrete substs.
                    (ty::ConstKind::Unevaluated(a_uv), ty::ConstKind::Unevaluated(b_uv)) => {
                        a_uv == b_uv
                    }
                    // FIXME(generic_const_exprs): We may want to either actually try
                    // to evaluate `a_ct` and `b_ct` if they are are fully concrete or something like
                    // this, for now we just return false here.
                    _ => false,
                }
            }
            (Node::Binop(a_op, al, ar), Node::Binop(b_op, bl, br)) if a_op == b_op => {
                self.try_unify(a.subtree(al), b.subtree(bl))
                    && self.try_unify(a.subtree(ar), b.subtree(br))
            }
            (Node::UnaryOp(a_op, av), Node::UnaryOp(b_op, bv)) if a_op == b_op => {
                self.try_unify(a.subtree(av), b.subtree(bv))
            }
            (Node::FunctionCall(a_f, a_args), Node::FunctionCall(b_f, b_args))
                if a_args.len() == b_args.len() =>
            {
                self.try_unify(a.subtree(a_f), b.subtree(b_f))
                    && iter::zip(a_args, b_args)
                        .all(|(&an, &bn)| self.try_unify(a.subtree(an), b.subtree(bn)))
            }
            (Node::Cast(a_kind, a_operand, a_ty), Node::Cast(b_kind, b_operand, b_ty))
                if (a_ty == b_ty) && (a_kind == b_kind) =>
            {
                self.try_unify(a.subtree(a_operand), b.subtree(b_operand))
            }
            // use this over `_ => false` to make adding variants to `Node` less error prone
            (Node::Cast(..), _)
            | (Node::FunctionCall(..), _)
            | (Node::UnaryOp(..), _)
            | (Node::Binop(..), _)
            | (Node::Leaf(..), _) => false,
        }
    }
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
