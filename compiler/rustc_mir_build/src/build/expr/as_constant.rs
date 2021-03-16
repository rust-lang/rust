//! See docs in build/expr/mod.rs

use crate::build::Builder;
use crate::thir::*;
use rustc_middle::mir::*;
use rustc_middle::ty::CanonicalUserTypeAnnotation;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a compile-time constant. Assumes that
    /// `expr` is a valid compile-time constant!
    crate fn as_constant(&mut self, expr: &Expr<'_, 'tcx>) -> Constant<'tcx> {
        let this = self;
        let Expr { ty, temp_lifetime: _, span, ref kind } = *expr;
        match *kind {
            ExprKind::Scope { region_scope: _, lint_level: _, value } => this.as_constant(value),
            ExprKind::Literal { literal, user_ty, const_id: _ } => {
                let user_ty = user_ty.map(|user_ty| {
                    this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                        span,
                        user_ty,
                        inferred_ty: ty,
                    })
                });
                assert_eq!(literal.ty, ty);
                Constant { span, user_ty, literal: literal.into() }
            }
            ExprKind::StaticRef { literal, .. } => {
                Constant { span, user_ty: None, literal: literal.into() }
            }
            ExprKind::ConstBlock { value } => {
                Constant { span: span, user_ty: None, literal: value.into() }
            }
            _ => span_bug!(span, "expression is not a valid constant {:?}", kind),
        }
    }
}
