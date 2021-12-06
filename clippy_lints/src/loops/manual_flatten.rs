use super::utils::make_iterator_snippet;
use super::MANUAL_FLATTEN;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{is_lang_ctor, path_to_local_id, peel_blocks_with_stmt};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionSome, ResultOk};
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
        let some_ctor = is_lang_ctor(cx, qpath, OptionSome);
        let ok_ctor = is_lang_ctor(cx, qpath, ResultOk);
        if some_ctor || ok_ctor;
        // Ensure expr in `if let` is not used afterwards
        if !is_local_used(cx, if_then, pat_hir_id);
        then {
            let if_let_type = if some_ctor { "Some" } else { "Ok" };
            // Prepare the error message
            let msg = format!("unnecessary `if let` since only the `{}` variant of the iterator element is used", if_let_type);

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

            span_lint_and_then(
                cx,
                MANUAL_FLATTEN,
                span,
                &msg,
                |diag| {
                    let sugg = format!("{}{}.flatten()", arg_snippet, copied);
                    diag.span_suggestion(
                        arg.span,
                        "try",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                    diag.span_help(
                        inner_expr.span,
                        "...and remove the `if let` statement in the for loop",
                    );
                }
            );
        }
    }
}
