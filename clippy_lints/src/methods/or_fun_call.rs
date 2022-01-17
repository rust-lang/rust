use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_lazy_eval;
use clippy_utils::source::{snippet, snippet_with_applicability, snippet_with_macro_callsite};
use clippy_utils::ty::{implements_trait, match_type};
use clippy_utils::{contains_return, is_trait_item, last_path_segment, paths};
use if_chain::if_chain;
use rustc_errors::emitter::MAX_SUGGESTION_HIGHLIGHT_LINES;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, sym};
use std::borrow::Cow;

use super::OR_FUN_CALL;

/// Checks for the `OR_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: &str,
    args: &'tcx [hir::Expr<'_>],
) {
    /// Checks for `unwrap_or(T::new())` or `unwrap_or(T::default())`.
    #[allow(clippy::too_many_arguments)]
    fn check_unwrap_or_default(
        cx: &LateContext<'_>,
        name: &str,
        fun: &hir::Expr<'_>,
        self_expr: &hir::Expr<'_>,
        arg: &hir::Expr<'_>,
        or_has_args: bool,
        span: Span,
        method_span: Span,
    ) -> bool {
        let is_default_default = || is_trait_item(cx, fun, sym::Default);

        let implements_default = |arg, default_trait_id| {
            let arg_ty = cx.typeck_results().expr_ty(arg);
            implements_trait(cx, arg_ty, default_trait_id, &[])
        };

        if_chain! {
            if !or_has_args;
            if name == "unwrap_or";
            if let hir::ExprKind::Path(ref qpath) = fun.kind;
            if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default);
            let path = last_path_segment(qpath).ident.name;
            // needs to target Default::default in particular or be *::new and have a Default impl
            // available
            if (matches!(path, kw::Default) && is_default_default())
                || (matches!(path, sym::new) && implements_default(arg, default_trait_id));

            then {
                let mut applicability = Applicability::MachineApplicable;
                let hint = "unwrap_or_default()";
                let mut sugg_span = span;

                let mut sugg: String = format!(
                    "{}.{}",
                    snippet_with_applicability(cx, self_expr.span, "..", &mut applicability),
                    hint
                );

                if sugg.lines().count() > MAX_SUGGESTION_HIGHLIGHT_LINES {
                    sugg_span = method_span.with_hi(span.hi());
                    sugg = hint.to_string();
                }

                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    sugg_span,
                    &format!("use of `{}` followed by a call to `{}`", name, path),
                    "try this",
                    sugg,
                    applicability,
                );

                true
            } else {
                false
            }
        }
    }

    /// Checks for `*or(foo())`.
    #[allow(clippy::too_many_arguments)]
    fn check_general_case<'tcx>(
        cx: &LateContext<'tcx>,
        name: &str,
        method_span: Span,
        self_expr: &hir::Expr<'_>,
        arg: &'tcx hir::Expr<'_>,
        span: Span,
        // None if lambda is required
        fun_span: Option<Span>,
    ) {
        // (path, fn_has_argument, methods, suffix)
        static KNOW_TYPES: [(&[&str], bool, &[&str], &str); 4] = [
            (&paths::BTREEMAP_ENTRY, false, &["or_insert"], "with"),
            (&paths::HASHMAP_ENTRY, false, &["or_insert"], "with"),
            (&paths::OPTION, false, &["map_or", "ok_or", "or", "unwrap_or"], "else"),
            (&paths::RESULT, true, &["or", "unwrap_or"], "else"),
        ];

        if_chain! {
            if KNOW_TYPES.iter().any(|k| k.2.contains(&name));

            if switch_to_lazy_eval(cx, arg);
            if !contains_return(arg);

            let self_ty = cx.typeck_results().expr_ty(self_expr);

            if let Some(&(_, fn_has_arguments, poss, suffix)) =
                KNOW_TYPES.iter().find(|&&i| match_type(cx, self_ty, i.0));

            if poss.contains(&name);

            then {
                let macro_expanded_snipped;
                let sugg: Cow<'_, str> = {
                    let (snippet_span, use_lambda) = match (fn_has_arguments, fun_span) {
                        (false, Some(fun_span)) => (fun_span, false),
                        _ => (arg.span, true),
                    };
                    let snippet = {
                        let not_macro_argument_snippet = snippet_with_macro_callsite(cx, snippet_span, "..");
                        if not_macro_argument_snippet == "vec![]" {
                            macro_expanded_snipped = snippet(cx, snippet_span, "..");
                            match macro_expanded_snipped.strip_prefix("$crate::vec::") {
                                Some(stripped) => Cow::from(stripped),
                                None => macro_expanded_snipped
                            }
                        }
                        else {
                            not_macro_argument_snippet
                        }
                    };

                    if use_lambda {
                        let l_arg = if fn_has_arguments { "_" } else { "" };
                        format!("|{}| {}", l_arg, snippet).into()
                    } else {
                        snippet
                    }
                };
                let span_replace_word = method_span.with_hi(span.hi());
                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    span_replace_word,
                    &format!("use of `{}` followed by a function call", name),
                    "try this",
                    format!("{}_{}({})", name, suffix, sugg),
                    Applicability::HasPlaceholders,
                );
            }
        }
    }

    if let [self_arg, arg] = args {
        let inner_arg = if let hir::ExprKind::Block(
            hir::Block {
                stmts: [],
                expr: Some(expr),
                ..
            },
            _,
        ) = arg.kind
        {
            expr
        } else {
            arg
        };
        match inner_arg.kind {
            hir::ExprKind::Call(fun, or_args) => {
                let or_has_args = !or_args.is_empty();
                if !check_unwrap_or_default(cx, name, fun, self_arg, arg, or_has_args, expr.span, method_span) {
                    let fun_span = if or_has_args { None } else { Some(fun.span) };
                    check_general_case(cx, name, method_span, self_arg, arg, expr.span, fun_span);
                }
            },
            hir::ExprKind::Index(..) | hir::ExprKind::MethodCall(..) => {
                check_general_case(cx, name, method_span, self_arg, arg, expr.span, None);
            },
            _ => (),
        }
    }
}
