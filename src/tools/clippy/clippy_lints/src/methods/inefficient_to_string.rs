use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_type_lang_item, walk_ptrs_ty_depth};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::{Symbol, sym};

use super::INEFFICIENT_TO_STRING;

/// Checks for the `INEFFICIENT_TO_STRING` lint
pub fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    method_name: Symbol,
    receiver: &hir::Expr<'_>,
    args: &[hir::Expr<'_>],
) {
    if args.is_empty()
        && method_name == sym::to_string
        && let Some(to_string_meth_did) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && cx.tcx.is_diagnostic_item(sym::to_string_method, to_string_meth_did)
        && let Some(args) = cx.typeck_results().node_args_opt(expr.hir_id)
        && let arg_ty = cx.typeck_results().expr_ty_adjusted(receiver)
        && let self_ty = args.type_at(0)
        && let (deref_self_ty, deref_count) = walk_ptrs_ty_depth(self_ty)
        && deref_count >= 1
        && specializes_tostring(cx, deref_self_ty)
    {
        span_lint_and_then(
            cx,
            INEFFICIENT_TO_STRING,
            expr.span,
            format!("calling `to_string` on `{arg_ty}`"),
            |diag| {
                diag.help(format!(
                    "`{self_ty}` implements `ToString` through a slower blanket impl, but `{deref_self_ty}` has a fast specialization of `ToString`"
                ));
                let mut applicability = Applicability::MachineApplicable;
                let arg_snippet = snippet_with_applicability(cx, receiver.span, "..", &mut applicability);
                diag.span_suggestion(
                    expr.span,
                    "try dereferencing the receiver",
                    format!("({}{arg_snippet}).to_string()", "*".repeat(deref_count)),
                    applicability,
                );
            },
        );
    }
}

/// Returns whether `ty` specializes `ToString`.
/// Currently, these are `str`, `String`, and `Cow<'_, str>`.
fn specializes_tostring(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    if let ty::Str = ty.kind() {
        return true;
    }

    if is_type_lang_item(cx, ty, hir::LangItem::String) {
        return true;
    }

    if let ty::Adt(adt, args) = ty.kind() {
        cx.tcx.is_diagnostic_item(sym::Cow, adt.did()) && args.type_at(1).is_str()
    } else {
        false
    }
}
