use crate::mir::interpret::ConstValue;
use crate::mir::interpret::{LitToConstInput, Scalar};
use crate::ty::{
    self, InlineConstSubsts, InlineConstSubstsParts, InternalSubsts, ParamEnv, ParamEnvAnd, Ty,
    TyCtxt, TypeFoldable,
};
use rustc_data_structures::intern::Interned;
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_macros::HashStable;
use std::fmt;

mod int;
mod kind;
mod valtree;

pub use int::*;
pub use kind::*;
pub use valtree::*;

/// Use this rather than `ConstS`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[cfg_attr(not(bootstrap), rustc_pass_by_value)]
pub struct Const<'tcx>(pub Interned<'tcx, ConstS<'tcx>>);

impl<'tcx> fmt::Debug for Const<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This reflects what `Const` looked liked before `Interned` was
        // introduced. We print it like this to avoid having to update expected
        // output in a lot of tests.
        write!(f, "Const {{ ty: {:?}, val: {:?} }}", self.ty(), self.val())
    }
}

/// Typed constant value.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, HashStable, TyEncodable, TyDecodable)]
pub struct ConstS<'tcx> {
    pub ty: Ty<'tcx>,
    pub val: ConstKind<'tcx>,
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstS<'_>, 48);

impl<'tcx> Const<'tcx> {
    pub fn ty(self) -> Ty<'tcx> {
        self.0.ty
    }

    pub fn val(self) -> ConstKind<'tcx> {
        self.0.val
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
        debug!("Const::from_anon_const(def={:?})", def);

        let body_id = match tcx.hir().get_by_def_id(def.did) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def.did.to_def_id()),
                "from_anon_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        let ty = tcx.type_of(def.def_id_for_type_of());

        match Self::try_eval_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => tcx.mk_const(ty::ConstS {
                val: ty::ConstKind::Unevaluated(ty::Unevaluated {
                    def: def.to_global(),
                    substs: InternalSubsts::identity_for_item(tcx, def.did.to_def_id()),
                    promoted: None,
                }),
                ty,
            }),
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

        use hir::{def::DefKind::ConstParam, def::Res, ExprKind, Path, QPath};
        match expr.kind {
            ExprKind::Path(QPath::Resolved(_, &Path { res: Res::Def(ConstParam, def_id), .. })) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
                let item_id = tcx.hir().get_parent_node(hir_id);
                let item_def_id = tcx.hir().local_def_id(item_id);
                let generics = tcx.generics_of(item_def_id.to_def_id());
                let index = generics.param_def_id_to_index[&def_id];
                let name = tcx.hir().name(hir_id);
                Some(tcx.mk_const(ty::ConstS {
                    val: ty::ConstKind::Param(ty::ParamConst::new(index, name)),
                    ty,
                }))
            }
            _ => None,
        }
    }

    pub fn from_inline_const(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        debug!("Const::from_inline_const(def_id={:?})", def_id);

        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

        let body_id = match tcx.hir().get(hir_id) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => span_bug!(
                tcx.def_span(def_id.to_def_id()),
                "from_inline_const can only process anonymous constants"
            ),
        };

        let expr = &tcx.hir().body(body_id).value;

        let ty = tcx.typeck(def_id).node_type(hir_id);

        let ret = match Self::try_eval_lit_or_param(tcx, ty, expr) {
            Some(v) => v,
            None => {
                let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id());
                let parent_substs =
                    tcx.erase_regions(InternalSubsts::identity_for_item(tcx, typeck_root_def_id));
                let substs =
                    InlineConstSubsts::new(tcx, InlineConstSubstsParts { parent_substs, ty })
                        .substs;
                tcx.mk_const(ty::ConstS {
                    val: ty::ConstKind::Unevaluated(ty::Unevaluated {
                        def: ty::WithOptConstParam::unknown(def_id).to_global(),
                        substs,
                        promoted: None,
                    }),
                    ty,
                })
            }
        };
        debug_assert!(!ret.has_free_regions());
        ret
    }

    /// Interns the given value as a constant.
    #[inline]
    pub fn from_value(tcx: TyCtxt<'tcx>, val: ConstValue<'tcx>, ty: Ty<'tcx>) -> Self {
        tcx.mk_const(ConstS { val: ConstKind::Value(val), ty })
    }

    #[inline]
    /// Interns the given scalar as a constant.
    pub fn from_scalar(tcx: TyCtxt<'tcx>, val: Scalar, ty: Ty<'tcx>) -> Self {
        Self::from_value(tcx, ConstValue::Scalar(val), ty)
    }

    #[inline]
    /// Creates a constant with the given integer value and interns it.
    pub fn from_bits(tcx: TyCtxt<'tcx>, bits: u128, ty: ParamEnvAnd<'tcx, Ty<'tcx>>) -> Self {
        let size = tcx
            .layout_of(ty)
            .unwrap_or_else(|e| panic!("could not compute layout for {:?}: {:?}", ty, e))
            .size;
        Self::from_scalar(tcx, Scalar::from_uint(bits, size), ty.value)
    }

    #[inline]
    /// Creates an interned zst constant.
    pub fn zero_sized(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        Self::from_scalar(tcx, Scalar::ZST, ty)
    }

    #[inline]
    /// Creates an interned bool constant.
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        Self::from_bits(tcx, v as u128, ParamEnv::empty().and(tcx.types.bool))
    }

    #[inline]
    /// Creates an interned usize constant.
    pub fn from_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
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
        self.val().eval(tcx, param_env).try_to_bits(size)
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<bool> {
        self.val().eval(tcx, param_env).try_to_bool()
    }

    #[inline]
    pub fn try_eval_usize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Option<u64> {
        self.val().eval(tcx, param_env).try_to_machine_usize(tcx)
    }

    #[inline]
    /// Tries to evaluate the constant if it is `Unevaluated`. If that doesn't succeed, return the
    /// unevaluated constant.
    pub fn eval(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Const<'tcx> {
        if let Some(val) = self.val().try_eval(tcx, param_env) {
            match val {
                Ok(val) => Const::from_value(tcx, val, self.ty()),
                Err(ErrorReported) => tcx.const_error(self.ty()),
            }
        } else {
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
    pub fn eval_usize(self, tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> u64 {
        self.try_eval_usize(tcx, param_env)
            .unwrap_or_else(|| bug!("expected usize, got {:#?}", self))
    }
}

pub fn const_param_default<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Const<'tcx> {
    let default_def_id = match tcx.hir().get_by_def_id(def_id.expect_local()) {
        hir::Node::GenericParam(hir::GenericParam {
            kind: hir::GenericParamKind::Const { ty: _, default: Some(ac) },
            ..
        }) => tcx.hir().local_def_id(ac.hir_id),
        _ => span_bug!(
            tcx.def_span(def_id),
            "`const_param_default` expected a generic parameter with a constant"
        ),
    };
    Const::from_anon_const(tcx, default_def_id)
}
