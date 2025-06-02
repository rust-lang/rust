use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::{FN_TO_NUMERIC_CAST_WITH_TRUNCATION, utils};

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // We only want to check casts to `ty::Uint` or `ty::Int`
    let Some(to_nbits) = utils::int_ty_to_nbits(cx.tcx, cast_to) else {
        return;
    };
    match cast_from.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => {
            let mut applicability = Applicability::MaybeIncorrect;
            let from_snippet = snippet_with_applicability(cx, cast_expr.span, "x", &mut applicability);

            if to_nbits < cx.tcx.data_layout.pointer_size.bits() {
                span_lint_and_sugg(
                    cx,
                    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
                    expr.span,
                    format!("casting function pointer `{from_snippet}` to `{cast_to}`, which truncates the value"),
                    "try",
                    format!("{from_snippet} as usize"),
                    applicability,
                );
            }
        },
        _ => {},
    }
}
