use super::MANUAL_FLATTEN;
use super::utils::make_iterator_snippet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{HasSession, indent_of, reindent_multiline, snippet_with_applicability};
use clippy_utils::visitors::is_local_used;
use clippy_utils::{higher, is_refutable, path_to_local_id, peel_blocks_with_stmt, span_contains_comment};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

/// Check for unnecessary `if let` usage in a for loop where only the `Some` or `Ok` variant of the
/// iterator element is used.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    span: Span,
    msrv: Msrv,
) {
    let inner_expr = peel_blocks_with_stmt(body);
    if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else: None, .. })
            = higher::IfLet::hir(cx, inner_expr)
        // Ensure match_expr in `if let` statement is the same as the pat from the for-loop
        && let PatKind::Binding(_, pat_hir_id, _, _) = pat.kind
        && path_to_local_id(let_expr, pat_hir_id)
        // Ensure the `if let` statement is for the `Some` variant of `Option` or the `Ok` variant of `Result`
        && let PatKind::TupleStruct(ref qpath, [inner_pat], _) = let_pat.kind
        && let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(qpath, let_pat.hir_id)
        && let Some(variant_id) = cx.tcx.opt_parent(ctor_id)
        && let some_ctor = cx.tcx.lang_items().option_some_variant() == Some(variant_id)
        && let ok_ctor = cx.tcx.lang_items().result_ok_variant() == Some(variant_id)
        && (some_ctor || ok_ctor)
        // Ensure expr in `if let` is not used afterwards
        && !is_local_used(cx, if_then, pat_hir_id)
        && msrv.meets(cx, msrvs::ITER_FLATTEN)
        && !is_refutable(cx, inner_pat)
    {
        if arg.span.from_expansion() || if_then.span.from_expansion() {
            return;
        }
        let if_let_type = if some_ctor { "Some" } else { "Ok" };
        // Prepare the error message
        let msg =
            format!("unnecessary `if let` since only the `{if_let_type}` variant of the iterator element is used");

        // Prepare the help message
        let mut applicability = if span_contains_comment(cx.sess().source_map(), body.span) {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        };
        let arg_snippet = make_iterator_snippet(cx, arg, &mut applicability);
        let copied = match cx.typeck_results().expr_ty(let_expr).kind() {
            ty::Ref(_, inner, _) => match inner.kind() {
                ty::Ref(..) => ".copied()",
                _ => "",
            },
            _ => "",
        };

        let help_msg = "try `.flatten()` and remove the `if let` statement in the for loop";

        let pat_snippet =
            snippet_with_applicability(cx, inner_pat.span.source_callsite(), "_", &mut applicability).to_string();
        let body_snippet =
            snippet_with_applicability(cx, if_then.span.source_callsite(), "[body]", &mut applicability).to_string();
        let suggestions = vec![
            // flatten the iterator
            (arg.span, format!("{arg_snippet}{copied}.flatten()")),
            (pat.span, pat_snippet),
            // remove the `if let` statement
            (
                body.span,
                reindent_multiline(&body_snippet, true, indent_of(cx, body.span)),
            ),
        ];

        span_lint_and_then(cx, MANUAL_FLATTEN, span, msg, |diag| {
            diag.span_help(inner_expr.span, help_msg);
            diag.multipart_suggestion("try", suggestions, applicability);
        });
    }
}
