use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_lazy_eval;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{expr_type_is_certain, implements_trait, is_type_diagnostic_item};
use clippy_utils::{contains_return, is_default_equivalent, is_default_equivalent_call, last_path_segment};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Span;
use rustc_span::symbol::{self, sym, Symbol};
use {rustc_ast as ast, rustc_hir as hir};

use super::{OR_FUN_CALL, UNWRAP_OR_DEFAULT};

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
    /// Similarly checks for `unwrap_or_else(T::new)`, `unwrap_or_else(T::default)`,
    /// `or_insert_with(T::new)` or `or_insert_with(T::default)`.
    #[allow(clippy::too_many_arguments)]
    fn check_unwrap_or_default(
        cx: &LateContext<'_>,
        name: &str,
        receiver: &hir::Expr<'_>,
        fun: &hir::Expr<'_>,
        call_expr: Option<&hir::Expr<'_>>,
        span: Span,
        method_span: Span,
    ) -> bool {
        if !expr_type_is_certain(cx, receiver) {
            return false;
        }

        let is_new = |fun: &hir::Expr<'_>| {
            if let hir::ExprKind::Path(ref qpath) = fun.kind {
                let path = last_path_segment(qpath).ident.name;
                matches!(path, sym::new)
            } else {
                false
            }
        };

        let output_type_implements_default = |fun| {
            let fun_ty = cx.typeck_results().expr_ty(fun);
            if let ty::FnDef(def_id, args) = fun_ty.kind() {
                let output_ty = cx.tcx.fn_sig(def_id).instantiate(cx.tcx, args).skip_binder().output();
                cx.tcx
                    .get_diagnostic_item(sym::Default)
                    .map_or(false, |default_trait_id| {
                        implements_trait(cx, output_ty, default_trait_id, &[])
                    })
            } else {
                false
            }
        };

        let sugg = match (name, call_expr.is_some()) {
            ("unwrap_or", true) | ("unwrap_or_else", false) => sym!(unwrap_or_default),
            ("or_insert", true) | ("or_insert_with", false) => sym!(or_default),
            _ => return false,
        };

        let receiver_ty = cx.typeck_results().expr_ty_adjusted(receiver).peel_refs();
        let has_suggested_method = receiver_ty.ty_adt_def().is_some_and(|adt_def| {
            cx.tcx
                .inherent_impls(adt_def.did())
                .iter()
                .flat_map(|impl_id| cx.tcx.associated_items(impl_id).filter_by_name_unhygienic(sugg))
                .any(|assoc| {
                    assoc.fn_has_self_parameter
                        && cx.tcx.fn_sig(assoc.def_id).skip_binder().inputs().skip_binder().len() == 1
                })
        });
        if !has_suggested_method {
            return false;
        }

        // needs to target Default::default in particular or be *::new and have a Default impl
        // available
        if (is_new(fun) && output_type_implements_default(fun))
            || match call_expr {
                Some(call_expr) => is_default_equivalent(cx, call_expr),
                None => is_default_equivalent_call(cx, fun) || closure_body_returns_empty_to_string(cx, fun),
            }
        {
            span_lint_and_sugg(
                cx,
                UNWRAP_OR_DEFAULT,
                method_span.with_hi(span.hi()),
                &format!("use of `{name}` to construct default value"),
                "try",
                format!("{sugg}()"),
                Applicability::MachineApplicable,
            );

            true
        } else {
            false
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
                    "try",
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
                if or_has_args
                    || !check_unwrap_or_default(cx, name, receiver, fun, Some(inner_arg), expr.span, method_span)
                {
                    let fun_span = if or_has_args { None } else { Some(fun.span) };
                    check_general_case(cx, name, method_span, receiver, arg, None, expr.span, fun_span);
                }
            },
            hir::ExprKind::Path(..) | hir::ExprKind::Closure(..) => {
                check_unwrap_or_default(cx, name, receiver, inner_arg, None, expr.span, method_span);
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

fn closure_body_returns_empty_to_string(cx: &LateContext<'_>, e: &hir::Expr<'_>) -> bool {
    if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = e.kind {
        let body = cx.tcx.hir().body(body);

        if body.params.is_empty()
            && let hir::Expr{ kind, .. } = &body.value
            && let hir::ExprKind::MethodCall(hir::PathSegment {ident, ..}, self_arg, _, _) = kind
            && ident.name == sym::to_string
            && let hir::Expr{ kind, .. } = self_arg
            && let hir::ExprKind::Lit(lit) = kind
            && let ast::LitKind::Str(symbol::kw::Empty, _) = lit.node
        {
            return true;
        }
    }

    false
}
