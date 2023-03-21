use crate::middle::resolve_bound_vars as rbv;
use crate::mir::interpret::LitToConstInput;
use crate::ty::{self, InternalSubsts, ParamEnv, ParamEnvAnd, Ty, TyCtxt};
use rustc_data_structures::intern::Interned;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_macros::HashStable;
use std::fmt;

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
pub use valtree::*;

/// Use this rather than `ConstData`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Const<'tcx>(pub(super) Interned<'tcx, ConstData<'tcx>>);

impl<'tcx> fmt::Debug for Const<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This reflects what `Const` looked liked before `Interned` was
        // introduced. We print it like this to avoid having to update expected
        // output in a lot of tests.
        write!(f, "Const {{ ty: {:?}, kind: {:?} }}", self.ty(), self.kind())
    }
}

/// Typed constant value.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, TyEncodable, TyDecodable)]
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

    /// Literals and const generic parameters are eagerly converted to a constant, everything else
    /// becomes `Unevaluated`.
    pub fn from_anon_const(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        Self::from_opt_const_arg_anon_const(tcx, ty::WithOptConstParam::unknown(def_id))
    }

    #[instrument(skip(tcx), level = "debug")]
    pub fn from_opt_const_arg_anon_const(
        tcx: TyCtxt<'tcx>,
        def: ty::WithOptConstParam<LocalDefId>,
    ) -> Self {
        let body_id = match tcx.hir().get_by_def_id(def.did) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def.did.to_def_id()),
                "from_anon_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        let ty = tcx
            .type_of(def.def_id_for_type_of())
            .no_bound_vars()
            .expect("const parameter types cannot be generic");

        match Self::try_eval_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => tcx.mk_const(
                ty::UnevaluatedConst {
                    def: def.to_global(),
                    substs: InternalSubsts::identity_for_item(tcx, def.did),
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
                        &format!("Const::from_anon_const: couldn't lit_to_const {:?}", e),
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
                        Some(tcx.mk_const(ty::ParamConst::new(index, name), param_ty))
                    }
                    Some(rbv::ResolvedArg::LateBound(debruijn, index, _)) => Some(tcx.mk_const(
                        ty::ConstKind::Bound(debruijn, ty::BoundVar::from_u32(index)),
                        param_ty,
                    )),
                    Some(rbv::ResolvedArg::Error(guar)) => {
                        Some(tcx.const_error_with_guaranteed(param_ty, guar))
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
        tcx.mk_const(
            ty::ValTree::from_scalar_int(ScalarInt::try_from_uint(bits, size).unwrap()),
            ty.value,
        )
    }

    #[inline]
    /// Creates an interned zst constant.
    pub fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        tcx.mk_const(ty::ValTree::zst(), ty)
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
        self.kind().eval(tcx, param_env).try_to_bits(size)
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<bool> {
        self.kind().eval(tcx, param_env).try_to_bool()
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Option<u64> {
        self.kind().eval(tcx, param_env).try_to_target_usize(tcx)
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that doesn't succeed, return the
    /// unevaluated constant.
    pub fn eval(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Const<'tcx> {
        if let Some(val) = self.kind().try_eval_for_typeck(tcx, param_env) {
            match val {
                Ok(val) => tcx.mk_const(val, self.ty()),
                Err(guar) => tcx.const_error_with_guaranteed(self.ty(), guar),
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
    ty::EarlyBinder(Const::from_anon_const(tcx, default_def_id))
}
