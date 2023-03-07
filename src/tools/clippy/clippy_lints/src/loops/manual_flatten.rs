use super::utils::make_iterator_snippet;
use super::MANUAL_FLATTEN;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{path_to_local_id, peel_blocks_with_stmt};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Span;

/// Check for unnecessary `if let` usage in a for loop where only the `Some` or `Ok` variant of the
/// iterator element is used.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    span: Span,
) {
    let inner_expr = peel_blocks_with_stmt(body);
    if_chain! {
        if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else: None })
            = higher::IfLet::hir(cx, inner_expr);
        // Ensure match_expr in `if let` statement is the same as the pat from the for-loop
        if let PatKind::Binding(_, pat_hir_id, _, _) = pat.kind;
        if path_to_local_id(let_expr, pat_hir_id);
        // Ensure the `if let` statement is for the `Some` variant of `Option` or the `Ok` variant of `Result`
        if let PatKind::TupleStruct(ref qpath, _, _) = let_pat.kind;
        if let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(qpath, let_pat.hir_id);
        if let Some(variant_id) = cx.tcx.opt_parent(ctor_id);
        let some_ctor = cx.tcx.lang_items().option_some_variant() == Some(variant_id);
        let ok_ctor = cx.tcx.lang_items().result_ok_variant() == Some(variant_id);
        if some_ctor || ok_ctor;
        // Ensure expr in `if let` is not used afterwards
        if !is_local_used(cx, if_then, pat_hir_id);
        then {
            let if_let_type = if some_ctor { "Some" } else { "Ok" };
            // Prepare the error message
            let msg = format!("unnecessary `if let` since only the `{if_let_type}` variant of the iterator element is used");

            // Prepare the help message
            let mut applicability = Applicability::MaybeIncorrect;
            let arg_snippet = make_iterator_snippet(cx, arg, &mut applicability);
            let copied = match cx.typeck_results().expr_ty(let_expr).kind() {
                ty::Ref(_, inner, _) => match inner.kind() {
                    ty::Ref(..) => ".copied()",
                    _ => ""
                }
                _ => ""
            };

            let sugg = format!("{arg_snippet}{copied}.flatten()");

            // If suggestion is not a one-liner, it won't be shown inline within the error message. In that case,
            // it will be shown in the extra `help` message at the end, which is why the first `help_msg` needs
            // to refer to the correct relative position of the suggestion.
            let help_msg = if sugg.contains('\n') {
                "remove the `if let` statement in the for loop and then..."
            } else {
                "...and remove the `if let` statement in the for loop"
            };

            span_lint_and_then(
                cx,
                MANUAL_FLATTEN,
                span,
                &msg,
                |diag| {
                    diag.span_suggestion(
                        arg.span,
                        "try",
                        sugg,
                        applicability,
                    );
                    diag.span_help(
                        inner_expr.span,
                        help_msg,
                    );
                }
            );
        }
    }
}
