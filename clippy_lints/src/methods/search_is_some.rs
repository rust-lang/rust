use crate::utils::{
    is_type_diagnostic_item, match_trait_method, paths, snippet, snippet_with_applicability, span_lint_and_help,
    span_lint_and_sugg, strip_pat_refs,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;

use super::SEARCH_IS_SOME;

/// lint searching an Iterator followed by `is_some()`
/// or calling `find()` on a string followed by `is_some()`
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    search_method: &str,
    search_args: &'tcx [hir::Expr<'_>],
    is_some_args: &'tcx [hir::Expr<'_>],
    method_span: Span,
) {
    // lint if caller of search is an Iterator
    if match_trait_method(cx, &is_some_args[0], &paths::ITERATOR) {
        let msg = format!(
            "called `is_some()` after searching an `Iterator` with `{}`",
            search_method
        );
        let hint = "this is more succinctly expressed by calling `any()`";
        let search_snippet = snippet(cx, search_args[1].span, "..");
        if search_snippet.lines().count() <= 1 {
            // suggest `any(|x| ..)` instead of `any(|&x| ..)` for `find(|&x| ..).is_some()`
            // suggest `any(|..| *..)` instead of `any(|..| **..)` for `find(|..| **..).is_some()`
            let any_search_snippet = if_chain! {
                if search_method == "find";
                if let hir::ExprKind::Closure(_, _, body_id, ..) = search_args[1].kind;
                let closure_body = cx.tcx.hir().body(body_id);
                if let Some(closure_arg) = closure_body.params.get(0);
                then {
                    if let hir::PatKind::Ref(..) = closure_arg.pat.kind {
                        Some(search_snippet.replacen('&', "", 1))
                    } else if let PatKind::Binding(_, _, ident, _) = strip_pat_refs(&closure_arg.pat).kind {
                        let name = &*ident.name.as_str();
                        Some(search_snippet.replace(&format!("*{}", name), name))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // add note if not multi-line
            span_lint_and_sugg(
                cx,
                SEARCH_IS_SOME,
                method_span.with_hi(expr.span.hi()),
                &msg,
                "use `any()` instead",
                format!(
                    "any({})",
                    any_search_snippet.as_ref().map_or(&*search_snippet, String::as_str)
                ),
                Applicability::MachineApplicable,
            );
        } else {
            span_lint_and_help(cx, SEARCH_IS_SOME, expr.span, &msg, None, hint);
        }
    }
    // lint if `find()` is called by `String` or `&str`
    else if search_method == "find" {
        let is_string_or_str_slice = |e| {
            let self_ty = cx.typeck_results().expr_ty(e).peel_refs();
            if is_type_diagnostic_item(cx, self_ty, sym::string_type) {
                true
            } else {
                *self_ty.kind() == ty::Str
            }
        };
        if_chain! {
            if is_string_or_str_slice(&search_args[0]);
            if is_string_or_str_slice(&search_args[1]);
            then {
                let msg = "called `is_some()` after calling `find()` on a string";
                let mut applicability = Applicability::MachineApplicable;
                let find_arg = snippet_with_applicability(cx, search_args[1].span, "..", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    SEARCH_IS_SOME,
                    method_span.with_hi(expr.span.hi()),
                    msg,
                    "use `contains()` instead",
                    format!("contains({})", find_arg),
                    applicability,
                );
            }
        }
    }
}
