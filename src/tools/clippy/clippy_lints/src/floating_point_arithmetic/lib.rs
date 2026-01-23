use clippy_utils::sugg::Sugg;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;

// Adds type suffixes and parenthesis to method receivers if necessary
pub(super) fn prepare_receiver_sugg<'a>(
    cx: &LateContext<'_>,
    mut expr: &'a Expr<'a>,
    app: &mut Applicability,
) -> Sugg<'a> {
    let mut suggestion = Sugg::hir_with_applicability(cx, expr, "_", app);

    if let ExprKind::Unary(UnOp::Neg, inner_expr) = expr.kind {
        expr = inner_expr;
    }

    if let ty::Float(float_ty) = cx.typeck_results().expr_ty(expr).kind()
        // if the expression is a float literal and it is unsuffixed then
        // add a suffix so the suggestion is valid and unambiguous
        && let ExprKind::Lit(lit) = expr.kind
        && let ast::LitKind::Float(sym, ast::LitFloatType::Unsuffixed) = lit.node
    {
        let op = format!(
            "{suggestion}{}{}",
            // Check for float literals without numbers following the decimal
            // separator such as `2.` and adds a trailing zero
            if sym.as_str().ends_with('.') { "0" } else { "" },
            float_ty.name_str()
        )
        .into();

        suggestion = match suggestion {
            Sugg::MaybeParen(_) | Sugg::UnOp(UnOp::Neg, _) => Sugg::MaybeParen(op),
            _ => Sugg::NonParen(op),
        };
    }

    suggestion.maybe_paren()
}
