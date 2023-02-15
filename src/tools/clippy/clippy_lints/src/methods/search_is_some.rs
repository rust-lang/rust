use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::sugg::deref_closure_args;
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::{is_trait_method, strip_pat_refs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;

use super::SEARCH_IS_SOME;

/// lint searching an Iterator followed by `is_some()`
/// or calling `find()` on a string followed by `is_some()` or `is_none()`
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    expr: &'tcx hir::Expr<'_>,
    search_method: &str,
    is_some: bool,
    search_recv: &hir::Expr<'_>,
    search_arg: &'tcx hir::Expr<'_>,
    is_some_recv: &hir::Expr<'_>,
    method_span: Span,
) {
    let option_check_method = if is_some { "is_some" } else { "is_none" };
    // lint if caller of search is an Iterator
    if is_trait_method(cx, is_some_recv, sym::Iterator) {
        let msg = format!("called `{option_check_method}()` after searching an `Iterator` with `{search_method}`");
        let search_snippet = snippet(cx, search_arg.span, "..");
        if search_snippet.lines().count() <= 1 {
            // suggest `any(|x| ..)` instead of `any(|&x| ..)` for `find(|&x| ..).is_some()`
            // suggest `any(|..| *..)` instead of `any(|..| **..)` for `find(|..| **..).is_some()`
            let mut applicability = Applicability::MachineApplicable;
            let any_search_snippet = if_chain! {
                if search_method == "find";
                if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = search_arg.kind;
                let closure_body = cx.tcx.hir().body(body);
                if let Some(closure_arg) = closure_body.params.get(0);
                then {
                    if let hir::PatKind::Ref(..) = closure_arg.pat.kind {
                        Some(search_snippet.replacen('&', "", 1))
                    } else if let PatKind::Binding(..) = strip_pat_refs(closure_arg.pat).kind {
                        // `find()` provides a reference to the item, but `any` does not,
                        // so we should fix item usages for suggestion
                        if let Some(closure_sugg) = deref_closure_args(cx, search_arg) {
                            applicability = closure_sugg.applicability;
                            Some(closure_sugg.suggestion)
                        } else {
                            Some(search_snippet.to_string())
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            // add note if not multi-line
            if is_some {
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
                    applicability,
                );
            } else {
                let iter = snippet(cx, search_recv.span, "..");
                span_lint_and_sugg(
                    cx,
                    SEARCH_IS_SOME,
                    expr.span,
                    &msg,
                    "use `!_.any()` instead",
                    format!(
                        "!{iter}.any({})",
                        any_search_snippet.as_ref().map_or(&*search_snippet, String::as_str)
                    ),
                    applicability,
                );
            }
        } else {
            let hint = format!(
                "this is more succinctly expressed by calling `any()`{}",
                if option_check_method == "is_none" {
                    " with negation"
                } else {
                    ""
                }
            );
            span_lint_and_help(cx, SEARCH_IS_SOME, expr.span, &msg, None, &hint);
        }
    }
    // lint if `find()` is called by `String` or `&str`
    else if search_method == "find" {
        let is_string_or_str_slice = |e| {
            let self_ty = cx.typeck_results().expr_ty(e).peel_refs();
            if is_type_lang_item(cx, self_ty, hir::LangItem::String) {
                true
            } else {
                self_ty.is_str()
            }
        };
        if_chain! {
            if is_string_or_str_slice(search_recv);
            if is_string_or_str_slice(search_arg);
            then {
                let msg = format!("called `{option_check_method}()` after calling `find()` on a string");
                match option_check_method {
                    "is_some" => {
                        let mut applicability = Applicability::MachineApplicable;
                        let find_arg = snippet_with_applicability(cx, search_arg.span, "..", &mut applicability);
                        span_lint_and_sugg(
                            cx,
                            SEARCH_IS_SOME,
                            method_span.with_hi(expr.span.hi()),
                            &msg,
                            "use `contains()` instead",
                            format!("contains({find_arg})"),
                            applicability,
                        );
                    },
                    "is_none" => {
                        let string = snippet(cx, search_recv.span, "..");
                        let mut applicability = Applicability::MachineApplicable;
                        let find_arg = snippet_with_applicability(cx, search_arg.span, "..", &mut applicability);
                        span_lint_and_sugg(
                            cx,
                            SEARCH_IS_SOME,
                            expr.span,
                            &msg,
                            "use `!_.contains()` instead",
                            format!("!{string}.contains({find_arg})"),
                            applicability,
                        );
                    },
                    _ => (),
                }
            }
        }
    }
}
