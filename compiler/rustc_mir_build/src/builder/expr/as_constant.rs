//! See docs in builder/expr/mod.rs

use rustc_middle::mir;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::consts::lit_to_mir_constant;
use rustc_middle::ty::{
    self, CanonicalUserType, CanonicalUserTypeAnnotation, LitToConstInput, TyCtxt,
    UserTypeAnnotationIndex,
};
use rustc_span::span_bug;

use crate::builder::Builder;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    pub(crate) fn as_constant(&mut self, expr: &Expr<'tcx>) -> ConstOperand<'tcx> {
        let this = self; // See "LET_THIS_SELF".
        let tcx = this.tcx;
        let Expr { ty, temp_scope_id: _, span, ref kind } = *expr;
        match kind {
            ExprKind::Scope { region_scope: _, hir_id: _, value } => {
                this.as_constant(&this.thir[*value])
            }
            _ => as_constant_inner(
                expr,
                |user_ty| {
                    Some(this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span,
                        user_ty: user_ty.clone(),
                        inferred_ty: ty,
                    }))
                },
                tcx,
            ),
        }
    }
}

pub(crate) fn as_constant_inner<'tcx>(
    expr: &Expr<'tcx>,
    push_cuta: impl FnMut(&Box<CanonicalUserType<'tcx>>) -> Option<UserTypeAnnotationIndex>,
    tcx: TyCtxt<'tcx>,
) -> ConstOperand<'tcx> {
    let Expr { ty, temp_scope_id: _, span, ref kind } = *expr;

    match *kind {
        ExprKind::Literal { lit, neg } => {
            let const_ =
                lit_to_mir_constant(tcx, LitToConstInput { lit: lit.node, ty: Some(ty), neg });

            ConstOperand { span, user_ty: None, const_ }
        }
        ExprKind::NonHirLiteral { lit, ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);

            let const_ = Const::Val(ConstValue::Scalar(Scalar::Int(lit)), ty);

            ConstOperand { span, user_ty, const_ }
        }
        ExprKind::ZstLiteral { ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);

            let const_ = Const::Val(ConstValue::ZeroSized, ty);

            ConstOperand { span, user_ty, const_ }
        }
        ExprKind::NamedConst { def_id, args, ref user_ty } => {
            let user_ty = user_ty.as_ref().and_then(push_cuta);
            let uneval = mir::UnevaluatedConst::new(def_id, args);
            let const_ = Const::Unevaluated(uneval, ty);
            ConstOperand { user_ty, span, const_ }
        }
        ExprKind::ConstParam { param, def_id: _ } => {
            let const_param = ty::Const::new_param(tcx, param);
            let const_ = Const::Ty(expr.ty, const_param);

            ConstOperand { user_ty: None, span, const_ }
        }
        ExprKind::ConstBlock { did: def_id, args } => {
            let uneval = mir::UnevaluatedConst::new(def_id, args);
            let const_ = Const::Unevaluated(uneval, ty);

            ConstOperand { user_ty: None, span, const_ }
        }
        ExprKind::StaticRef { alloc_id, ty, .. } => {
            let const_val = ConstValue::Scalar(Scalar::from_pointer(alloc_id.into(), &tcx));
            let const_ = Const::Val(const_val, ty);

            ConstOperand { span, user_ty: None, const_ }
        }
        _ => span_bug!(span, "expression is not a valid constant {:?}", kind),
    }
}
