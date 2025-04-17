use super::TRANSMUTE_FLOAT_TO_INT;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg;
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_float_to_int` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    mut arg: &'tcx Expr<'_>,
    const_context: bool,
    msrv: Msrv,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Float(float_ty), ty::Int(_) | ty::Uint(_))
            if !const_context || msrv.meets(cx, msrvs::CONST_FLOAT_BITS_CONV) =>
        {
            span_lint_and_then(
                cx,
                TRANSMUTE_FLOAT_TO_INT,
                e.span,
                format!("transmute from a `{from_ty}` to a `{to_ty}`"),
                |diag| {
                    let mut sugg = sugg::Sugg::hir(cx, arg, "..");

                    if let ExprKind::Unary(UnOp::Neg, inner_expr) = &arg.kind {
                        arg = inner_expr;
                    }

                    if let ExprKind::Lit(lit) = &arg.kind
                        // if the expression is a float literal and it is unsuffixed then
                        // add a suffix so the suggestion is valid and unambiguous
                        && let ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) = lit.node
                    {
                        let op = format!("{sugg}{}", float_ty.name_str()).into();
                        match sugg {
                            sugg::Sugg::MaybeParen(_) => sugg = sugg::Sugg::MaybeParen(op),
                            _ => sugg = sugg::Sugg::NonParen(op),
                        }
                    }

                    sugg = sugg::Sugg::NonParen(format!("{}.to_bits()", sugg.maybe_paren()).into());

                    // cast the result of `to_bits` if `to_ty` is signed
                    sugg = if let ty::Int(int_ty) = to_ty.kind() {
                        sugg.as_ty(int_ty.name_str().to_string())
                    } else {
                        sugg
                    };

                    diag.span_suggestion(e.span, "consider using", sugg, Applicability::Unspecified);
                },
            );
            true
        },
        _ => false,
    }
}
