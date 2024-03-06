use crate::middle::resolve_bound_vars as rbv;
use crate::mir::interpret::{ErrorHandled, LitToConstInput, Scalar};
use crate::ty::{self, GenericArgs, ParamEnv, ParamEnvAnd, Ty, TyCtxt, TypeVisitableExt};
use rustc_data_structures::intern::Interned;
use rustc_error_messages::MultiSpan;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_macros::HashStable;
use rustc_type_ir::ConstKind as IrConstKind;
use rustc_type_ir::{ConstTy, IntoKind, TypeFlags, WithCachedTypeInfo};

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
use rustc_span::Span;
use rustc_span::DUMMY_SP;
pub use valtree::*;

pub type ConstKind<'tcx> = IrConstKind<TyCtxt<'tcx>>;

/// Use this rather than `ConstData`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, WithCachedTypeInfo<ConstData<'tcx>>>);

impl<'tcx> IntoKind for Const<'tcx> {
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

impl<'tcx> ConstTy<TyCtxt<'tcx>> for Const<'tcx> {
    fn ty(self) -> Ty<'tcx> {
        self.ty()
    }
}

/// Typed constant value.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct ConstData<'tcx> {
    pub ty: Ty<'tcx>,
    pub kind: ConstKind<'tcx>,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstData<'_>, 40);

impl<'tcx> Const<'tcx> {
    #[inline]
    pub fn ty(self) -> Ty<'tcx> {
        self.0.ty
    }

    #[inline]
    pub fn kind(self) -> ConstKind<'tcx> {
        self.0.kind
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
    pub fn new(tcx: TyCtxt<'tcx>, kind: ty::ConstKind<'tcx>, ty: Ty<'tcx>) -> Const<'tcx> {
        tcx.mk_ct_from_kind(kind, ty)
    }

    #[inline]
    pub fn new_param(tcx: TyCtxt<'tcx>, param: ty::ParamConst, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Param(param), ty)
    }

    #[inline]
    pub fn new_var(tcx: TyCtxt<'tcx>, infer: ty::ConstVid, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Var(infer)), ty)
    }

    #[inline]
    pub fn new_fresh(tcx: TyCtxt<'tcx>, fresh: u32, ty: Ty<'tcx>) -> Const<'tcx> {
        Const::new(tcx, ty::ConstKind::Infer(ty::InferConst::Fresh(fresh)), ty)
    }

    #[inline]
    pub fn new_infer(tcx: TyCtxt<'tcx>, infer: ty::InferConst, ty: Ty<'tcx>) -> Const<'tcx> {
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
        placeholder: ty::PlaceholderConst,
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
        let reported = tcx.dcx().span_delayed_bug(span, msg);
        Const::new_error(tcx, reported, ty)
    }
}

impl<'tcx> rustc_type_ir::new::Const<TyCtxt<'tcx>> for Const<'tcx> {
    fn new_anon_bound(
        tcx: TyCtxt<'tcx>,
        debruijn: ty::DebruijnIndex,
        var: ty::BoundVar,
        ty: Ty<'tcx>,
    ) -> Self {
        Const::new_bound(tcx, debruijn, var, ty)
    }
}

impl<'tcx> Const<'tcx> {
    /// Literals and const generic parameters are eagerly converted to a constant, everything else
    /// becomes `Unevaluated`.
    #[instrument(skip(tcx), level = "debug")]
    pub fn from_anon_const(tcx: TyCtxt<'tcx>, def: LocalDefId) -> Self {
        let body_id = match tcx.hir_node_by_def_id(def) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def.to_def_id()),
                "from_anon_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        let ty = tcx.type_of(def).no_bound_vars().expect("const parameter types cannot be generic");

        match Self::try_from_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => ty::Const::new_unevaluated(
                tcx,
                ty::UnevaluatedConst {
                    def: def.to_def_id(),
                    args: GenericArgs::identity_for_item(tcx, def.to_def_id()),
                },
                ty,
            ),
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
                Err(e) => {
                    tcx.dcx().span_delayed_bug(
                        expr.span,
                        format!("Const::from_anon_const: couldn't lit_to_const {e:?}"),
                    );
                }
            }
        }

        // FIXME(const_generics): We currently have to special case parameters because `min_const_generics`
        // does not provide the parents generics to anonymous constants. We still allow generic const
        // parameters by themselves however, e.g. `N`. These constants would cause an ICE if we were to
        // ever try to instantiate the generic parameters in their bodies.
        match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                _,
                &hir::Path { res: Res::Def(DefKind::ConstParam, def_id), .. },
            )) => {
                // Use the type from the param's definition, since we can resolve it,
                // not the expected parameter type from WithOptConstParam.
                let param_ty = tcx.type_of(def_id).instantiate_identity();
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

    /// Returns the evaluated constant
    #[inline]
    pub fn eval(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        span: Option<Span>,
    ) -> Result<ValTree<'tcx>, ErrorHandled> {
        assert!(!self.has_escaping_bound_vars(), "escaping vars in {self:?}");
        match self.kind() {
            ConstKind::Unevaluated(unevaluated) => {
                // FIXME(eddyb) maybe the `const_eval_*` methods should take
                // `ty::ParamEnvAnd` instead of having them separate.
                let (param_env, unevaluated) = unevaluated.prepare_for_eval(tcx, param_env);
                // try to resolve e.g. associated constants to their definition on an impl, and then
                // evaluate the const.
                let Some(c) = tcx.const_eval_resolve_for_typeck(param_env, unevaluated, span)?
                else {
                    // This can happen when we run on ill-typed code.
                    let e = tcx.dcx().span_delayed_bug(
                        span.unwrap_or(DUMMY_SP),
                        "`ty::Const::eval` called on a non-valtree-compatible type",
                    );
                    return Err(e.into());
                };
                Ok(c)
            }
            ConstKind::Value(val) => Ok(val),
            ConstKind::Error(g) => Err(g.into()),
            ConstKind::Param(_)
            | ConstKind::Infer(_)
            | ConstKind::Bound(_, _)
            | ConstKind::Placeholder(_)
            | ConstKind::Expr(_) => Err(ErrorHandled::TooGeneric(span.unwrap_or(DUMMY_SP))),
        }
    }

    /// Normalizes the constant to a value or an error if possible.
    #[inline]
    pub fn normalize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Self {
        match self.eval(tcx, param_env, None) {
            Ok(val) => Self::new_value(tcx, val, self.ty()),
            Err(ErrorHandled::Reported(r, _span)) => Self::new_error(tcx, r.into(), self.ty()),
            Err(ErrorHandled::TooGeneric(_span)) => self,
        }
    }

    #[inline]
    pub fn try_eval_scalar(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<Scalar> {
        self.eval(tcx, param_env, None).ok()?.try_to_scalar()
    }

    #[inline]
    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    pub fn try_eval_scalar_int(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<ScalarInt> {
        self.try_eval_scalar(tcx, param_env)?.try_to_int().ok()
    }

    #[inline]
    /// Attempts to evaluate the given constant to bits. Can fail to evaluate in the presence of
    /// generics (or erroneous code) or if the value can't be represented as bits (e.g. because it
    /// contains const generic parameters or pointers).
    pub fn try_eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<u128> {
        let int = self.try_eval_scalar_int(tcx, param_env)?;
        let size =
            tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(self.ty())).ok()?.size;
        // if `ty` does not depend on generic parameters, use an empty param_env
        int.to_bits(size).ok()
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env)
            .unwrap_or_else(|| bug!("expected bits of {:#?}, got {:#?}", self.ty(), self))
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<u64> {
        self.try_eval_scalar_int(tcx, param_env)?.try_to_target_usize(tcx).ok()
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<bool> {
        self.try_eval_scalar_int(tcx, param_env)?.try_into().ok()
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
            ty::ConstKind::Value(valtree) => valtree,
            _ => bug!("expected ConstKind::Value, got {:?}", self.kind()),
        }
    }

    /// Attempts to convert to a `ValTree`
    pub fn try_to_valtree(self) -> Option<ty::ValTree<'tcx>> {
        match self.kind() {
            ty::ConstKind::Value(valtree) => Some(valtree),
            _ => None,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        self.try_to_valtree()?.try_to_scalar()
    }

    #[inline]
    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_valtree()?.try_to_target_usize(tcx)
    }

    pub fn is_ct_infer(self) -> bool {
        matches!(self.kind(), ty::ConstKind::Infer(_))
    }
}

pub fn const_param_default(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::EarlyBinder<Const<'_>> {
    let default_def_id = match tcx.hir_node_by_def_id(def_id) {
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
