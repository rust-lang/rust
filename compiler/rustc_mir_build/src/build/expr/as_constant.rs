//! See docs in build/expr/mod.rs

use crate::build::{lit_to_mir_constant, Builder};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{ConstValue, LitToConstError, LitToConstInput, Scalar};
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, TyCtxt};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    crate fn as_constant(&mut self, expr: &Expr<'tcx>) -> Constant<'tcx> {
        let create_uneval_from_def_id =
            |tcx: TyCtxt<'tcx>, def_id: DefId, ty: Ty<'tcx>, substs: SubstsRef<'tcx>| {
                let uneval = ty::Unevaluated::new(ty::WithOptConstParam::unknown(def_id), substs);
                tcx.mk_const(ty::ConstS { val: ty::ConstKind::Unevaluated(uneval), ty })
            };

        let this = self;
        let tcx = this.tcx;
        let Expr { ty, temp_lifetime: _, span, ref kind } = *expr;
        match *kind {
            ExprKind::Scope { region_scope: _, lint_level: _, value } => {
                this.as_constant(&this.thir[value])
            }
            ExprKind::Literal { lit, neg } => {
                let literal =
                    match lit_to_mir_constant(tcx, LitToConstInput { lit: &lit.node, ty, neg }) {
                        Ok(c) => c,
                        Err(LitToConstError::Reported) => ConstantKind::Ty(tcx.const_error(ty)),
                        Err(LitToConstError::TypeError) => {
                            bug!("encountered type error in `lit_to_mir_constant")
                        }
                    };

                Constant { span, user_ty: None, literal }
            }
            ExprKind::NonHirLiteral { lit, user_ty } => {
                let user_ty = user_ty.map(|user_ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span,
                        user_ty,
                        inferred_ty: ty,
                    })
                });

                let literal = ConstantKind::Val(ConstValue::Scalar(Scalar::Int(lit)), ty);

                Constant { span, user_ty: user_ty, literal }
            }
            ExprKind::NamedConst { def_id, substs, user_ty } => {
                let user_ty = user_ty.map(|user_ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span,
                        user_ty,
                        inferred_ty: ty,
                    })
                });
                let literal = ConstantKind::Ty(create_uneval_from_def_id(tcx, def_id, ty, substs));

                Constant { user_ty, span, literal }
            }
            ExprKind::ConstParam { param, def_id: _ } => {
                let const_param =
                    tcx.mk_const(ty::ConstS { val: ty::ConstKind::Param(param), ty: expr.ty });
                let literal = ConstantKind::Ty(const_param);

                Constant { user_ty: None, span, literal }
            }
            ExprKind::ConstBlock { did: def_id, substs } => {
                let literal = ConstantKind::Ty(create_uneval_from_def_id(tcx, def_id, ty, substs));

                Constant { user_ty: None, span, literal }
            }
            ExprKind::StaticRef { alloc_id, ty, .. } => {
                let const_val = ConstValue::Scalar(Scalar::from_pointer(alloc_id.into(), &tcx));
                let literal = ConstantKind::Val(const_val, ty);

                Constant { span, user_ty: None, literal }
            }
            _ => span_bug!(span, "expression is not a valid constant {:?}", kind),
        }
    }
}
