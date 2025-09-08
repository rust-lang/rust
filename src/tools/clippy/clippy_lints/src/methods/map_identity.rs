use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item};
use clippy_utils::{is_expr_untyped_identity_function, is_mutable, is_trait_method, path_to_local_with_projections};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, ExprKind, Node, PatKind};
use rustc_lint::{LateContext, LintContext};
use rustc_span::{Span, Symbol, sym};

use super::MAP_IDENTITY;

const MSG: &str = "unnecessary map of the identity function";

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    caller: &hir::Expr<'_>,
    map_arg: &hir::Expr<'_>,
    name: Symbol,
    _map_span: Span,
) {
    let caller_ty = cx.typeck_results().expr_ty(caller);

    if (is_trait_method(cx, expr, sym::Iterator)
        || is_type_diagnostic_item(cx, caller_ty, sym::Result)
        || is_type_diagnostic_item(cx, caller_ty, sym::Option))
        && is_expr_untyped_identity_function(cx, map_arg)
        && let Some(call_span) = expr.span.trim_start(caller.span)
    {
        let main_sugg = (call_span, String::new());
        let mut app = if is_copy(cx, caller_ty) {
            // there is technically a behavioral change here for `Copy` iterators, where
            // `iter.map(|x| x).next()` would mutate a temporary copy of the iterator and
            // changing it to `iter.next()` mutates iter directly
            Applicability::Unspecified
        } else {
            Applicability::MachineApplicable
        };

        let needs_to_be_mutable = cx.typeck_results().expr_ty_adjusted(expr).is_mutable_ptr();
        if needs_to_be_mutable && !is_mutable(cx, caller) {
            if let Some(hir_id) = path_to_local_with_projections(caller)
                && let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
                && let PatKind::Binding(_, _, ident, _) = pat.kind
            {
                // We can reach the binding -- suggest making it mutable
                let suggs = vec![main_sugg, (ident.span.shrink_to_lo(), String::from("mut "))];

                let ident = snippet_with_applicability(cx.sess(), ident.span, "_", &mut app);

                span_lint_and_then(cx, MAP_IDENTITY, call_span, MSG, |diag| {
                    diag.multipart_suggestion(
                        format!("remove the call to `{name}`, and make `{ident}` mutable"),
                        suggs,
                        app,
                    );
                });
            } else {
                // If we can't make the binding mutable, prevent the suggestion from being automatically applied,
                // and add a complementary help message.
                app = Applicability::Unspecified;

                let method_requiring_mut = if let Node::Expr(expr) = cx.tcx.parent_hir_node(expr.hir_id)
                    && let ExprKind::MethodCall(method, ..) = expr.kind
                {
                    Some(method.ident)
                } else {
                    None
                };

                span_lint_and_then(cx, MAP_IDENTITY, call_span, MSG, |diag| {
                    diag.span_suggestion(main_sugg.0, format!("remove the call to `{name}`"), main_sugg.1, app);

                    let note = if let Some(method_requiring_mut) = method_requiring_mut {
                        format!("this must be made mutable to use `{method_requiring_mut}`")
                    } else {
                        "this must be made mutable".to_string()
                    };
                    diag.span_note(caller.span, note);
                });
            }
        } else {
            span_lint_and_sugg(
                cx,
                MAP_IDENTITY,
                main_sugg.0,
                MSG,
                format!("remove the call to `{name}`"),
                main_sugg.1,
                app,
            );
        }
    }
}
