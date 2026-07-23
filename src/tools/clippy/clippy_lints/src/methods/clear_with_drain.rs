use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef as _, MaybeResPath as _};
use clippy_utils::{is_full_collection_range, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, LangItem};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::CLEAR_WITH_DRAIN;

// Add `String` here when it is added to diagnostic items
const ACCEPTABLE_TYPES_WITH_ARG: [rustc_span::Symbol; 2] = [sym::Vec, sym::VecDeque];

const ACCEPTABLE_TYPES_WITHOUT_ARG: [rustc_span::Symbol; 3] = [sym::BinaryHeap, sym::HashMap, sym::HashSet];

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: Option<&Expr<'_>>) {
    if let Some(arg) = arg {
        if match_acceptable_type(cx, recv, &ACCEPTABLE_TYPES_WITH_ARG)
            && is_full_collection_range(cx, recv.res_local_id(), arg)
        {
            suggest(cx, expr, recv, span);
        }
    } else if match_acceptable_type(cx, recv, &ACCEPTABLE_TYPES_WITHOUT_ARG) {
        suggest(cx, expr, recv, span);
    }
}

fn match_acceptable_type(cx: &LateContext<'_>, expr: &Expr<'_>, types: &[rustc_span::Symbol]) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr).peel_refs();
    types.iter().any(|&ty| expr_ty.is_diag_item(cx, ty))
    // String type is a lang item but not a diagnostic item for now so we need a separate check
        || expr_ty.is_lang_item(cx, LangItem::String)
}

fn suggest(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span) {
    if let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
    // Use `opt_item_name` while `String` is not a diagnostic item
        && let Some(ty_name) = cx.tcx.opt_item_name(adt.did())
    {
        span_lint_and_sugg(
            cx,
            CLEAR_WITH_DRAIN,
            span.with_hi(expr.span.hi()),
            format!("`drain` used to clear a `{ty_name}`"),
            "try",
            "clear()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
