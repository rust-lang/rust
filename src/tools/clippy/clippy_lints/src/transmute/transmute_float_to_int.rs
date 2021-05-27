use super::TRANSMUTE_FLOAT_TO_INT;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use if_chain::if_chain;
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
    args: &'tcx [Expr<'_>],
    const_context: bool,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Float(float_ty), ty::Int(_) | ty::Uint(_)) if !const_context => {
            span_lint_and_then(
                cx,
                TRANSMUTE_FLOAT_TO_INT,
                e.span,
                &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                |diag| {
                    let mut expr = &args[0];
                    let mut arg = sugg::Sugg::hir(cx, expr, "..");

                    if let ExprKind::Unary(UnOp::Neg, inner_expr) = &expr.kind {
                        expr = inner_expr;
                    }

                    if_chain! {
                        // if the expression is a float literal and it is unsuffixed then
                        // add a suffix so the suggestion is valid and unambiguous
                        if let ExprKind::Lit(lit) = &expr.kind;
                        if let ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) = lit.node;
                        then {
                            let op = format!("{}{}", arg, float_ty.name_str()).into();
                            match arg {
                                sugg::Sugg::MaybeParen(_) => arg = sugg::Sugg::MaybeParen(op),
                                _ => arg = sugg::Sugg::NonParen(op)
                            }
                        }
                    }

                    arg = sugg::Sugg::NonParen(format!("{}.to_bits()", arg.maybe_par()).into());

                    // cast the result of `to_bits` if `to_ty` is signed
                    arg = if let ty::Int(int_ty) = to_ty.kind() {
                        arg.as_ty(int_ty.name_str().to_string())
                    } else {
                        arg
                    };

                    diag.span_suggestion(e.span, "consider using", arg.to_string(), Applicability::Unspecified);
                },
            );
            true
        },
        _ => false,
    }
}
