use either::Either;
use rustc_data_structures::intern::Interned;
use rustc_error_messages::MultiSpan;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir, HirId};
use rustc_macros::HashStable;
use rustc_type_ir::{self as ir, TypeFlags, WithCachedTypeInfo};
use tracing::{debug, instrument};

use crate::middle::resolve_bound_vars as rbv;
use crate::mir::interpret::{ErrorHandled, LitToConstInput, Scalar};
use crate::ty::{self, GenericArgs, ParamEnv, ParamEnvAnd, Ty, TyCtxt, TypeVisitableExt};

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
pub use valtree::*;

pub type ConstKind<'tcx> = ir::ConstKind<TyCtxt<'tcx>>;
pub type UnevaluatedConst<'tcx> = ir::UnevaluatedConst<TyCtxt<'tcx>>;

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ConstKind<'_>, 32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<ConstKind<'tcx>>>);

impl<'tcx> rustc_type_ir::inherent::IntoKind for Const<'tcx> {
    type Kind = ConstKind<'tcx>;

    fn kind(self) -> ConstKind<'tcx> {
        self.kind()
    }
}

impl<'tcx> rustc_type_ir::visit::Flags for Const<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<'tcx> Const<'tcx> {
    #[inline]
    pub fn kind(self) -> ConstKind<'tcx> {
        let a: &ConstKind<'tcx> = self.0.0;
        *a
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline]
    pub fn outer_exclusive_binder(self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }

    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>, kind: ty::ConstKind<'tcx>) -> Const<'tcx> {
        tcx.mk_ct_from_kind(kind)
    }

    #[inline]
    pub fn new_param(tcx: TyCtxt<'tcx>, param: ty::ParamConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Param(param))
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, infer: ty::ConstVid) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Var(infer)))
    }

    #[inline]
    pub fn new_fresh(tcx: TyCtxt<'tcx>, fresh: u32) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Fresh(fresh)))
    }

    #[inline]
    pub fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(infer))
    }

    #[inline]
    pub fn new_bound(
        tcx: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundVar,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Bound(debruijn, var))
    }

    #[inline]
    pub fn new_placeholder(tcx: TyCtxt<'tcx>, placeholder: ty::PlaceholderConst) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Placeholder(placeholder))
    }

    #[inline]
    pub fn new_unevaluated(tcx: TyCtxt<'tcx>, uv: ty::UnevaluatedConst<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Unevaluated(uv))
    }

    #[inline]
    pub fn new_value(tcx: TyCtxt<'tcx>, val: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Value(ty, val))
    }

    #[inline]
    pub fn new_expr(tcx: TyCtxt<'tcx>, expr: ty::Expr<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Expr(expr))
    }

    #[inline]
    pub fn new_error(tcx: TyCtxt<'tcx>, e: ty::ErrorGuaranteed) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Error(e))
    }

    /// Like [Ty::new_error] but for constants.
    #[track_caller]
    pub fn new_misc_error(tcx: TyCtxt<'tcx>) -> Const<'tcx> {
        Const::new_error_with_message(
            tcx,
            DUMMY_SP,
            "ty::ConstKind::Error constructed but no error reported",
        )
    }

    /// Like [Ty::new_error_with_message] but for constants.
    #[track_caller]
    pub fn new_error_with_message<S: Into<MultiSpan>>(
        tcx: TyCtxt<'tcx>,
        span: S,
        msg: &'static str,
    ) -> Const<'tcx> {
        let reported = tcx.dcx().span_delayed_bug(span, msg);
        Const::new_error(tcx, reported)
    }
}

impl<'tcx> rustc_type_ir::inherent::Const<TyCtxt<'tcx>> for Const<'tcx> {
    fn try_to_target_usize(self, interner: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_target_usize(interner)
    }

    fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst) -> Self {
        Const::new_infer(tcx, infer)
    }

    fn new_var(tcx: TyCtxt<'tcx>, vid: ty::ConstVid) -> Self {
        Const::new_var(tcx, vid)
    }

    fn new_bound(interner: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Const::new_bound(interner, debruijn, var)
    }

    fn new_anon_bound(tcx: TyCtxt<'tcx>, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        Const::new_bound(tcx, debruijn, var)
    }

    fn new_unevaluated(interner: TyCtxt<'tcx>, uv: ty::UnevaluatedConst<'tcx>) -> Self {
        Const::new_unevaluated(interner, uv)
    }

    fn new_expr(interner: TyCtxt<'tcx>, expr: ty::Expr<'tcx>) -> Self {
        Const::new_expr(interner, expr)
    }

    fn new_error(interner: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> Self {
        Const::new_error(interner, guar)
    }
}

/// In some cases, [`hir::ConstArg`]s that are being used in the type system
/// through const generics need to have their type "fed" to them
/// using the query system.
///
/// Use this enum with [`Const::from_const_arg`] to instruct it with the
/// desired behavior.
#[derive(Debug, Clone, Copy)]
pub enum FeedConstTy {
    /// Feed the type.
    ///
    /// The `DefId` belongs to the const param that we are supplying
    /// this (anon) const arg to.
    Param(DefId),
    /// Don't feed the type.
    No,
}

impl<'tcx> Const<'tcx> {
    /// Convert a [`hir::ConstArg`] to a [`ty::Const`](Self).
    #[instrument(skip(tcx), level = "debug")]
    pub fn from_const_arg(
        tcx: TyCtxt<'tcx>,
        const_arg: &'tcx hir::ConstArg<'tcx>,
        feed: FeedConstTy,
    ) -> Self {
        if let FeedConstTy::Param(param_def_id) = feed
            && let hir::ConstArgKind::Anon(anon) = &const_arg.kind
        {
            tcx.feed_anon_const_type(anon.def_id, tcx.type_of(param_def_id));
        }

        match const_arg.kind {
            hir::ConstArgKind::Path(qpath) => {
                // FIXME(min_generic_const_args): for now only params are lowered to ConstArgKind::Path
                Self::from_param(tcx, qpath, const_arg.hir_id)
            }
            hir::ConstArgKind::Anon(anon) => Self::from_anon_const(tcx, anon.def_id),
        }
    }

    /// Literals and const generic parameters are eagerly converted to a constant, everything else
    /// becomes `Unevaluated`.
    #[instrument(skip(tcx), level = "debug")]
    pub fn from_anon_const(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Self {
        let body_id = match tcx.hir_node_by_def_id(def) {
            hir::Node::AnonConst(ac) => ac.body,
            node => span_bug!(
                tcx.def_span(def.to_def_id()),
                "from_anon_const can only process anonymous constants, not {node:?}"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        let ty = tcx.type_of(def).no_bound_vars().expect("const parameter types cannot be generic");

        match Self::try_from_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => ty::Const::new_unevaluated(tcx, ty::UnevaluatedConst {
                def: def.to_def_id(),
                args: GenericArgs::identity_for_item(tcx, def.to_def_id()),
            }),
        }
    }

    /// Lower a const param to a [`Const`].
    ///
    /// IMPORTANT: `qpath` must be a const param, otherwise this will panic
    fn from_param(tcx: TyCtxt<'tcx>, qpath: hir::QPath<'tcx>, hir_id: HirId) -> Self {
        let hir::QPath::Resolved(_, &hir::Path { res: Res::Def(DefKind::ConstParam, def_id), .. }) =
            qpath
        else {
            span_bug!(qpath.span(), "non-param {qpath:?} passed to Const::from_param")
        };

        match tcx.named_bound_var(hir_id) {
            Some(rbv::ResolvedArg::EarlyBound(_)) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let item_def_id = tcx.parent(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                let name = tcx.item_name(def_id);
                ty::Const::new_param(tcx, ty::ParamConst::new(index, name))
            }
            Some(rbv::ResolvedArg::LateBound(debruijn, index, _)) => {
                ty::Const::new_bound(tcx, debruijn, ty::BoundVar::from_u32(index))
            }
            Some(rbv::ResolvedArg::Error(guar)) => ty::Const::new_error(tcx, guar),
            arg => bug!("unexpected bound var resolution for {:?}: {arg:?}", hir_id),
        }
    }

    #[instrument(skip(tcx), level = "debug")]
    fn try_from_lit_or_param(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Option<Self> {
        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = match &expr.kind {
            hir::ExprKind::Block(block, _) if block.stmts.is_empty() && block.expr.is_some() => {
                block.expr.as_ref().unwrap()
            }
            _ => expr,
        };

        if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            &hir::Path { res: Res::Def(DefKind::ConstParam, _), .. },
        )) = expr.kind
        {
            span_bug!(expr.span, "try_from_lit: received const param which shouldn't be possible");
        };

        let lit_input = match expr.kind {
            hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: false }),
            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => match expr.kind {
                hir::ExprKind::Lit(lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: true }),
                _ => None,
            },
            _ => None,
        };

        if let Some(lit_input) = lit_input {
            // If an error occurred, ignore that it's a literal and leave reporting the error up to
            // mir.
            match tcx.at(expr.span).lit_to_const(lit_input) {
                Ok(c) => return Some(c),
                Err(_) if lit_input.ty.has_aliases() => {
                    // allow the `ty` to be an alias type, though we cannot handle it here
                    return None;
                }
                Err(e) => {
                    tcx.dcx().span_delayed_bug(
                        expr.span,
                        format!("Const::from_anon_const: couldn't lit_to_const {e:?}"),
                    );
                }
            }
        }

        None
    }

    #[inline]
    /// Creates a constant with the given integer value and interns it.
    pub fn from_bits(tcx: TyCtxt<'tcx>, bits: u128, ty: ParamEnvAnd<'tcx, Ty<'tcx>>) -> Self {
        let size = tcx
            .layout_of(ty)
            .unwrap_or_else(|e| panic!("could not compute layout for {ty:?}: {e:?}"))
            .size;
        ty::Const::new_value(
            tcx,
            ty::ValTree::from_scalar_int(ScalarInt::try_from_uint(bits, size).unwrap()),
            ty.value,
        )
    }

    #[inline]
    /// Creates an interned zst constant.
    pub fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        ty::Const::new_value(tcx, ty::ValTree::zst(), ty)
    }

    #[inline]
    /// Creates an interned bool constant.
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        Self::from_bits(tcx, v as u128, ParamEnv::empty().and(tcx.types.bool))
    }

    #[inline]
    /// Creates an interned usize constant.
    pub fn from_target_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        Self::from_bits(tcx, n as u128, ParamEnv::empty().and(tcx.types.usize))
    }

    /// Returns the evaluated constant as a valtree;
    /// if that fails due to a valtree-incompatible type, indicate which type that is
    /// by returning `Err(Left(bad_type))`.
    #[inline]
    pub fn eval_valtree(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        span: Span,
    ) -> Result<(Ty<'tcx>, ValTree<'tcx>), Either<Ty<'tcx>, ErrorHandled>> {
        assert!(!self.has_escaping_bound_vars(), "escaping vars in {self:?}");
        match self.kind() {
            ConstKind::Unevaluated(unevaluated) => {
                // FIXME(eddyb) maybe the `const_eval_*` methods should take
                // `ty::ParamEnvAnd` instead of having them separate.
                let (param_env, unevaluated) = unevaluated.prepare_for_eval(tcx, param_env);
                // try to resolve e.g. associated constants to their definition on an impl, and then
                // evaluate the const.
                match tcx.const_eval_resolve_for_typeck(param_env, unevaluated, span) {
                    Ok(Ok(c)) => {
                        Ok((tcx.type_of(unevaluated.def).instantiate(tcx, unevaluated.args), c))
                    }
                    Ok(Err(bad_ty)) => Err(Either::Left(bad_ty)),
                    Err(err) => Err(Either::Right(err)),
                }
            }
            ConstKind::Value(ty, val) => Ok((ty, val)),
            ConstKind::Error(g) => Err(Either::Right(g.into())),
            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(_, _)
            | ConstKind::Placeholder(_)
            | ConstKind::Expr(_) => Err(Either::Right(ErrorHandled::TooGeneric(span))),
        }
    }

    /// Returns the evaluated constant
    #[inline]
    pub fn eval(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        span: Span,
    ) -> Result<(Ty<'tcx>, ValTree<'tcx>), ErrorHandled> {
        self.eval_valtree(tcx, param_env, span).map_err(|err| {
            match err {
                Either::Right(err) => err,
                Either::Left(_bad_ty) => {
                    // This can happen when we run on ill-typed code.
                    let e = tcx.dcx().span_delayed_bug(
                        span,
                        "`ty::Const::eval` called on a non-valtree-compatible type",
                    );
                    e.into()
                }
            }
        })
    }

    /// Normalizes the constant to a value or an error if possible.
    #[inline]
    pub fn normalize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Self {
        match self.eval(tcx, param_env, DUMMY_SP) {
            Ok((ty, val)) => Self::new_value(tcx, val, ty),
            Err(ErrorHandled::Reported(r, _span)) => Self::new_error(tcx, r.into()),
            Err(ErrorHandled::TooGeneric(_span)) => self,
        }
    }

    #[inline]
    pub fn try_eval_scalar(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<(Ty<'tcx>, Scalar)> {
        let (ty, val) = self.eval(tcx, param_env, DUMMY_SP).ok()?;
        let val = val.try_to_scalar()?;
        Some((ty, val))
    }

    #[inline]
    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    pub fn try_eval_scalar_int(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<(Ty<'tcx>, ScalarInt)> {
        let (ty, scalar) = self.try_eval_scalar(tcx, param_env)?;
        let val = scalar.try_to_scalar_int().ok()?;
        Some((ty, val))
    }

    #[inline]
    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    pub fn try_eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<u128> {
        let (ty, scalar) = self.try_eval_scalar_int(tcx, param_env)?;
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
        // if `ty` does not depend on generic parameters, use an empty param_env
        Some(scalar.to_bits(size))
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env)
            .unwrap_or_else(|| bug!("failed to evalate {:#?} to bits", self))
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<u64> {
        let (_, scalar) = self.try_eval_scalar_int(tcx, param_env)?;
        Some(scalar.to_target_usize(tcx))
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<bool> {
        let (_, scalar) = self.try_eval_scalar_int(tcx, param_env)?;
        scalar.try_into().ok()
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid `usize`.
    pub fn eval_target_usize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> u64 {
        self.try_eval_target_usize(tcx, param_env)
            .unwrap_or_else(|| bug!("expected usize, got {:#?}", self))
    }

    /// Panics if self.kind != ty::ConstKind::Value
    pub fn to_valtree(self) -> ty::ValTree<'tcx> {
        match self.kind() {
            ty::ConstKind::Value(_, valtree) => valtree,
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    /// Attempts to convert to a `ValTree`
    pub fn try_to_valtree(self) -> Option<ty::ValTree<'tcx>> {
        match self.kind() {
            ty::ConstKind::Value(_, valtree) => Some(valtree),
            _ => None,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        self.try_to_valtree()?.try_to_scalar()
    }

    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_valtree()?.try_to_scalar_int()?.try_to_bool().ok()
    }

    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_valtree()?.try_to_target_usize(tcx)
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(_))
    }
}

pub fn const_param_default<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'tcx, Const<'tcx>> {
    let default_ct = match tcx.hir_node_by_def_id(def_id) {
        hir::Node::GenericParam(hir::GenericParam {
            kind: hir::GenericParamKind::Const { default: Some(ct), .. },
            ..
        }) => ct,
        _ => span_bug!(
            tcx.def_span(def_id),
            "`const_param_default` expected a generic parameter with a constant"
        ),
    };
    ty::EarlyBinder::bind(Const::from_const_arg(tcx, default_ct, FeedConstTy::No))
}
