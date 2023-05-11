use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_lazy_eval;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::{contains_return, is_trait_item, last_path_segment};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, sym, Symbol};

use super::OR_FUN_CALL;

/// Checks for the `OR_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: &str,
    receiver: &'tcx hir::Expr<'_>,
    args: &'tcx [hir::Expr<'_>],
) {
    /// Checks for `unwrap_or(T::new())`, `unwrap_or(T::default())`,
    /// `or_insert(T::new())` or `or_insert(T::default())`.
    #[allow(clippy::too_many_arguments)]
    fn check_unwrap_or_default(
        cx: &LateContext<'_>,
        name: &str,
        fun: &hir::Expr<'_>,
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
            if let Some(sugg) = match name {
                "unwrap_or" => Some("unwrap_or_default"),
                "or_insert" => Some("or_default"),
                _ => None,
            };
            if let hir::ExprKind::Path(ref qpath) = fun.kind;
            if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default);
            let path = last_path_segment(qpath).ident.name;
            // needs to target Default::default in particular or be *::new and have a Default impl
            // available
            if (matches!(path, kw::Default) && is_default_default())
                || (matches!(path, sym::new) && implements_default(arg, default_trait_id));

            then {
                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    method_span.with_hi(span.hi()),
                    &format!("use of `{name}` followed by a call to `{path}`"),
                    "try this",
                    format!("{sugg}()"),
                    Applicability::MachineApplicable,
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
        // `Some` if fn has second argument
        second_arg: Option<&hir::Expr<'_>>,
        span: Span,
        // None if lambda is required
        fun_span: Option<Span>,
    ) {
        // (path, fn_has_argument, methods, suffix)
        const KNOW_TYPES: [(Symbol, bool, &[&str], &str); 4] = [
            (sym::BTreeEntry, false, &["or_insert"], "with"),
            (sym::HashMapEntry, false, &["or_insert"], "with"),
            (sym::Option, false, &["map_or", "ok_or", "or", "unwrap_or"], "else"),
            (sym::Result, true, &["or", "unwrap_or"], "else"),
        ];

        if_chain! {
            if KNOW_TYPES.iter().any(|k| k.2.contains(&name));

            if switch_to_lazy_eval(cx, arg);
            if !contains_return(arg);

            let self_ty = cx.typeck_results().expr_ty(self_expr);

            if let Some(&(_, fn_has_arguments, poss, suffix)) =
                KNOW_TYPES.iter().find(|&&i| is_type_diagnostic_item(cx, self_ty, i.0));

            if poss.contains(&name);

            then {
                let ctxt = span.ctxt();
                let mut app = Applicability::HasPlaceholders;
                let sugg = {
                    let (snippet_span, use_lambda) = match (fn_has_arguments, fun_span) {
                        (false, Some(fun_span)) => (fun_span, false),
                        _ => (arg.span, true),
                    };

                    let snip = snippet_with_context(cx, snippet_span, ctxt, "..", &mut app).0;
                    let snip = if use_lambda {
                        let l_arg = if fn_has_arguments { "_" } else { "" };
                        format!("|{l_arg}| {snip}")
                    } else {
                        snip.into_owned()
                    };

                    if let Some(f) = second_arg {
                        let f = snippet_with_context(cx, f.span, ctxt, "..", &mut app).0;
                        format!("{snip}, {f}")
                    } else {
                        snip
                    }
                };
                let span_replace_word = method_span.with_hi(span.hi());
                span_lint_and_sugg(
                    cx,
                    OR_FUN_CALL,
                    span_replace_word,
                    &format!("use of `{name}` followed by a function call"),
                    "try this",
                    format!("{name}_{suffix}({sugg})"),
                    app,
                );
            }
        }
    }

    let extract_inner_arg = |arg: &'tcx hir::Expr<'_>| {
        if let hir::ExprKind::Block(
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
        }
    };

    if let [arg] = args {
        let inner_arg = extract_inner_arg(arg);
        match inner_arg.kind {
            hir::ExprKind::Call(fun, or_args) => {
                let or_has_args = !or_args.is_empty();
                if !check_unwrap_or_default(cx, name, fun, arg, or_has_args, expr.span, method_span) {
                    let fun_span = if or_has_args { None } else { Some(fun.span) };
                    check_general_case(cx, name, method_span, receiver, arg, None, expr.span, fun_span);
                }
            },
            hir::ExprKind::Index(..) | hir::ExprKind::MethodCall(..) => {
                check_general_case(cx, name, method_span, receiver, arg, None, expr.span, None);
            },
            _ => (),
        }
    }

    // `map_or` takes two arguments
    if let [arg, lambda] = args {
        let inner_arg = extract_inner_arg(arg);
        if let hir::ExprKind::Call(fun, or_args) = inner_arg.kind {
            let fun_span = if or_args.is_empty() { Some(fun.span) } else { None };
            check_general_case(cx, name, method_span, receiver, arg, Some(lambda), expr.span, fun_span);
        }
    }
}
