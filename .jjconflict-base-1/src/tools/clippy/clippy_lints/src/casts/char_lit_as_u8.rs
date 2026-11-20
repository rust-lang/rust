use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

use super::CHAR_LIT_AS_U8;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from_expr: &Expr<'_>, cast_to: Ty<'_>) {
    if let ExprKind::Lit(l) = &cast_from_expr.kind
        && let LitKind::Char(c) = l.node
        && ty::Uint(UintTy::U8) == *cast_to.kind()
    {
        let mut applicability = Applicability::MachineApplicable;
        let snippet = snippet_with_applicability(cx, cast_from_expr.span, "'x'", &mut applicability);

        span_lint_and_then(
            cx,
            CHAR_LIT_AS_U8,
            expr.span,
            "casting a character literal to `u8` truncates",
            |diag| {
                diag.note("`char` is four bytes wide, but `u8` is a single byte");

                if c.is_ascii() {
                    diag.span_suggestion(
                        expr.span,
                        "use a byte literal instead",
                        format!("b{snippet}"),
                        applicability,
                    );
                }
            },
        );
    }
}
