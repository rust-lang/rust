use crate::middle::resolve_bound_vars as rbv;
use crate::mir::interpret::{AllocId, ConstValue, LitToConstInput, Scalar};
use crate::ty::{self, InternalSubsts, ParamEnv, ParamEnvAnd, Ty, TyCtxt, TypeVisitableExt};
use rustc_data_structures::intern::Interned;
use rustc_error_messages::MultiSpan;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_macros::HashStable;

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
use rustc_span::ErrorGuaranteed;
use rustc_span::DUMMY_SP;
use rustc_target::abi::Size;
pub use valtree::*;

use super::sty::ConstKind;

/// Use this rather than `ConstData`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, ConstData<'tcx>>);

/// Typed constant value.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, TyEncodable, TyDecodable)]
pub struct ConstData<'tcx> {
    pub ty: Ty<'tcx>,
    pub kind: ConstKind<'tcx>,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstData<'_>, 40);

enum EvalMode {
    Typeck,
    Mir,
}

enum EvalResult<'tcx> {
    ValTree(ty::ValTree<'tcx>),
    ConstVal(ConstValue<'tcx>),
}

impl<'tcx> Const<'tcx> {
    #[inline]
    pub fn ty(self) -> Ty<'tcx> {
        self.0.ty
    }

    #[inline]
    pub fn kind(self) -> ConstKind<'tcx> {
        self.0.kind.clone()
    }

    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>, kind: ty::ConstKind<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        tcx.mk_ct_from_kind(kind, ty)
    }

    #[inline]
    pub fn new_param(tcx: TyCtxt<'tcx>, param: ty::ParamConst, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Param(param), ty)
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, infer: ty::ConstVid<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Var(infer)), ty)
    }

    #[inline]
    pub fn new_fresh(tcx: TyCtxt<'tcx>, fresh: u32, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Fresh(fresh)), ty)
    }

    #[inline]
    pub fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(infer), ty)
    }

    #[inline]
    pub fn new_bound(
        tcx: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundVar,
        ty: Ty<'tcx>,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Bound(debruijn, var), ty)
    }

    #[inline]
    pub fn new_placeholder(
        tcx: TyCtxt<'tcx>,
        placeholder: ty::PlaceholderConst<'tcx>,
        ty: Ty<'tcx>,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Placeholder(placeholder), ty)
    }

    #[inline]
    pub fn new_unevaluated(
        tcx: TyCtxt<'tcx>,
        uv: ty::UnevaluatedConst<'tcx>,
        ty: Ty<'tcx>,
    ) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Unevaluated(uv), ty)
    }

    #[inline]
    pub fn new_value(tcx: TyCtxt<'tcx>, val: ty::ValTree<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Value(val), ty)
    }

    #[inline]
    pub fn new_expr(tcx: TyCtxt<'tcx>, expr: ty::Expr<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Expr(expr), ty)
    }

    #[inline]
    pub fn new_error(tcx: TyCtxt<'tcx>, e: ty::ErrorGuaranteed, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Error(e), ty)
    }

    /// Like [Ty::new_error] but for constants.
    #[track_caller]
    pub fn new_misc_error(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new_error_with_message(
            tcx,
            ty,
            DUMMY_SP,
            "ty::ConstKind::Error constructed but no error reported",
        )
    }

    /// Like [Ty::new_error_with_message] but for constants.
    #[track_caller]
    pub fn new_error_with_message<S: Into<MultiSpan>>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        span: S,
        msg: &'static str,
    ) -> Const<'tcx> {
        let reported = tcx.sess.delay_span_bug(span, msg);
        Const::new_error(tcx, reported, ty)
    }

    /// Literals and const generic parameters are eagerly converted to a constant, everything else
    /// becomes `Unevaluated`.
    #[instrument(skip(tcx), level = "debug")]
    pub fn from_anon_const(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Self {
        let body_id = match tcx.hir().get_by_def_id(def) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def.to_def_id()),
                "from_anon_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        let ty = tcx.type_of(def).no_bound_vars().expect("const parameter types cannot be generic");

        match Self::try_eval_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => ty::Const::new_unevaluated(
                tcx,
                ty::UnevaluatedConst {
                    def: def.to_def_id(),
                    substs: InternalSubsts::identity_for_item(tcx, def.to_def_id()),
                },
                ty,
            ),
        }
    }

    #[instrument(skip(tcx), level = "debug")]
    fn try_eval_lit_or_param(
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

        let lit_input = match expr.kind {
            hir::ExprKind::Lit(ref lit) => Some(LitToConstInput { lit: &lit.node, ty, neg: false }),
            hir::ExprKind::Unary(hir::UnOp::Neg, ref expr) => match expr.kind {
                hir::ExprKind::Lit(ref lit) => {
                    Some(LitToConstInput { lit: &lit.node, ty, neg: true })
                }
                _ => None,
            },
            _ => None,
        };

        if let Some(lit_input) = lit_input {
            // If an error occurred, ignore that it's a literal and leave reporting the error up to
            // mir.
            match tcx.at(expr.span).lit_to_const(lit_input) {
                Ok(c) => return Some(c),
                Err(e) => {
                    tcx.sess.delay_span_bug(
                        expr.span,
                        format!("Const::from_anon_const: couldn't lit_to_const {:?}", e),
                    );
                }
            }
        }

        match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                _,
                &hir::Path { res: Res::Def(DefKind::ConstParam, def_id), .. },
            )) => {
                // Use the type from the param's definition, since we can resolve it,
                // not the expected parameter type from WithOptConstParam.
                let param_ty = tcx.type_of(def_id).subst_identity();
                match tcx.named_bound_var(expr.hir_id) {
                    Some(rbv::ResolvedArg::EarlyBound(_)) => {
                        // Find the name and index of the const parameter by indexing the generics of
                        // the parent item and construct a `ParamConst`.
                        let item_def_id = tcx.parent(def_id);
                        let generics = tcx.generics_of(item_def_id);
                        let index = generics.param_def_id_to_index[&def_id];
                        let name = tcx.item_name(def_id);
                        Some(ty::Const::new_param(tcx, ty::ParamConst::new(index, name), param_ty))
                    }
                    Some(rbv::ResolvedArg::LateBound(debruijn, index, _)) => {
                        Some(ty::Const::new_bound(
                            tcx,
                            debruijn,
                            ty::BoundVar::from_u32(index),
                            param_ty,
                        ))
                    }
                    Some(rbv::ResolvedArg::Error(guar)) => {
                        Some(ty::Const::new_error(tcx, guar, param_ty))
                    }
                    arg => bug!("unexpected bound var resolution for {:?}: {arg:?}", expr.hir_id),
                }
            }
            _ => None,
        }
    }

    /// Panics if self.kind != ty::ConstKind::Value
    pub fn to_valtree(self) -> ty::ValTree<'tcx> {
        match self.kind() {
            ty::ConstKind::Value(valtree) => valtree,
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    #[inline]
    /// Creates a constant with the given integer value and interns it.
    pub fn from_bits(tcx: TyCtxt<'tcx>, bits: u128, ty: ParamEnvAnd<'tcx, Ty<'tcx>>) -> Self {
        let size = tcx
            .layout_of(ty)
            .unwrap_or_else(|e| panic!("could not compute layout for {:?}: {:?}", ty, e))
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

    #[inline]
    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    pub fn try_eval_bits(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        assert_eq!(self.ty(), ty);
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
        // if `ty` does not depend on generic parameters, use an empty param_env
        self.eval(tcx, param_env).try_to_bits(size)
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<bool> {
        self.eval(tcx, param_env).try_to_bool()
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<u64> {
        self.eval(tcx, param_env).try_to_target_usize(tcx)
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that doesn't succeed, return the
    /// unevaluated constant.
    pub fn eval(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Const<'tcx> {
        if let Some(val) = self.try_eval_for_typeck(tcx, param_env) {
            match val {
                Ok(val) => ty::Const::new_value(tcx, val, self.ty()),
                Err(guar) => ty::Const::new_error(tcx, guar, self.ty()),
            }
        } else {
            // Either the constant isn't evaluatable or ValTree creation failed.
            self
        }
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>, ty: Ty<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env, ty)
            .unwrap_or_else(|| bug!("expected bits of {:#?}, got {:#?}", ty, self))
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid `usize`.
    pub fn eval_target_usize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> u64 {
        self.try_eval_target_usize(tcx, param_env)
            .unwrap_or_else(|| bug!("expected usize, got {:#?}", self))
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that isn't possible or necessary
    /// return `None`.
    // FIXME(@lcnr): Completely rework the evaluation/normalization system for `ty::Const` once valtrees are merged.
    pub fn try_eval_for_mir(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<Result<ConstValue<'tcx>, ErrorGuaranteed>> {
        match self.try_eval_inner(tcx, param_env, EvalMode::Mir) {
            Some(Ok(EvalResult::ValTree(_))) => unreachable!(),
            Some(Ok(EvalResult::ConstVal(v))) => Some(Ok(v)),
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that isn't possible or necessary
    /// return `None`.
    // FIXME(@lcnr): Completely rework the evaluation/normalization system for `ty::Const` once valtrees are merged.
    pub fn try_eval_for_typeck(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<Result<ty::ValTree<'tcx>, ErrorGuaranteed>> {
        match self.try_eval_inner(tcx, param_env, EvalMode::Typeck) {
            Some(Ok(EvalResult::ValTree(v))) => Some(Ok(v)),
            Some(Ok(EvalResult::ConstVal(_))) => unreachable!(),
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }

    #[inline]
    fn try_eval_inner(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        eval_mode: EvalMode,
    ) -> Option<Result<EvalResult<'tcx>, ErrorGuaranteed>> {
        assert!(!self.has_escaping_bound_vars(), "escaping vars in {self:?}");
        if let ConstKind::Unevaluated(unevaluated) = self.kind() {
            use crate::mir::interpret::ErrorHandled;

            // HACK(eddyb) this erases lifetimes even though `const_eval_resolve`
            // also does later, but we want to do it before checking for
            // inference variables.
            // Note that we erase regions *before* calling `with_reveal_all_normalized`,
            // so that we don't try to invoke this query with
            // any region variables.

            // HACK(eddyb) when the query key would contain inference variables,
            // attempt using identity substs and `ParamEnv` instead, that will succeed
            // when the expression doesn't depend on any parameters.
            // FIXME(eddyb, skinny121) pass `InferCtxt` into here when it's available, so that
            // we can call `infcx.const_eval_resolve` which handles inference variables.
            let param_env_and = if (param_env, unevaluated).has_non_region_infer() {
                tcx.param_env(unevaluated.def).and(ty::UnevaluatedConst {
                    def: unevaluated.def,
                    substs: InternalSubsts::identity_for_item(tcx, unevaluated.def),
                })
            } else {
                tcx.erase_regions(param_env)
                    .with_reveal_all_normalized(tcx)
                    .and(tcx.erase_regions(unevaluated))
            };

            // FIXME(eddyb) maybe the `const_eval_*` methods should take
            // `ty::ParamEnvAnd` instead of having them separate.
            let (param_env, unevaluated) = param_env_and.into_parts();
            // try to resolve e.g. associated constants to their definition on an impl, and then
            // evaluate the const.
            match eval_mode {
                EvalMode::Typeck => {
                    match tcx.const_eval_resolve_for_typeck(param_env, unevaluated, None) {
                        // NOTE(eddyb) `val` contains no lifetimes/types/consts,
                        // and we use the original type, so nothing from `substs`
                        // (which may be identity substs, see above),
                        // can leak through `val` into the const we return.
                        Ok(val) => Some(Ok(EvalResult::ValTree(val?))),
                        Err(ErrorHandled::TooGeneric) => None,
                        Err(ErrorHandled::Reported(e)) => Some(Err(e.into())),
                    }
                }
                EvalMode::Mir => {
                    match tcx.const_eval_resolve(param_env, unevaluated.expand(), None) {
                        // NOTE(eddyb) `val` contains no lifetimes/types/consts,
                        // and we use the original type, so nothing from `substs`
                        // (which may be identity substs, see above),
                        // can leak through `val` into the const we return.
                        Ok(val) => Some(Ok(EvalResult::ConstVal(val))),
                        Err(ErrorHandled::TooGeneric) => None,
                        Err(ErrorHandled::Reported(e)) => Some(Err(e.into())),
                    }
                }
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn try_to_value(self) -> Option<ty::ValTree<'tcx>> {
        if let ConstKind::Value(val) = self.kind() { Some(val) } else { None }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar<AllocId>> {
        self.try_to_value()?.try_to_scalar()
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        self.try_to_value()?.try_to_scalar_int()
    }

    #[inline]
    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_value()?.try_to_target_usize(tcx)
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(_))
    }
}

pub fn const_param_default(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::EarlyBinder<Const<'_>> {
    let default_def_id = match tcx.hir().get_by_def_id(def_id) {
        hir::Node::GenericParam(hir::GenericParam {
            kind: hir::GenericParamKind::Const { default: Some(ac), .. },
            ..
        }) => ac.def_id,
        _ => span_bug!(
            tcx.def_span(def_id),
            "`const_param_default` expected a generic parameter with a constant"
        ),
    };
    ty::EarlyBinder::bind(Const::from_anon_const(tcx, default_def_id))
}
