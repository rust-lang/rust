use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::{has_non_owning_mutable_access, implements_trait};
use clippy_utils::{is_mutable, is_trait_method, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Instance;
use rustc_span::{Span, sym};

use super::DOUBLE_ENDED_ITERATOR_LAST;

pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, self_expr: &'_ Expr<'_>, call_span: Span) {
    let typeck = cx.typeck_results();

    // if the "last" method is that of Iterator
    if is_trait_method(cx, expr, sym::Iterator)
        // if self implements DoubleEndedIterator
        && let Some(deiter_id) = cx.tcx.get_diagnostic_item(sym::DoubleEndedIterator)
        && let self_type = cx.typeck_results().expr_ty(self_expr)
        && implements_trait(cx, self_type.peel_refs(), deiter_id, &[])
        // resolve the method definition
        && let id = typeck.type_dependent_def_id(expr.hir_id).unwrap()
        && let args = typeck.node_args(expr.hir_id)
        && let Ok(Some(fn_def)) = Instance::try_resolve(cx.tcx, cx.typing_env(), id, args)
        // find the provided definition of Iterator::last
        && let Some(item) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(last_def) = cx.tcx.provided_trait_methods(item).find(|m| m.name().as_str() == "last")
        // if the resolved method is the same as the provided definition
        && fn_def.def_id() == last_def.def_id
        && let self_ty = cx.typeck_results().expr_ty(self_expr)
        && !has_non_owning_mutable_access(cx, self_ty)
    {
        let mut sugg = vec![(call_span, String::from("next_back()"))];
        let mut dont_apply = false;

        // if `self_expr` is a reference, it is mutable because it is used for `.last()`
        // TODO: Change this to lint only when the referred iterator is not used later. If it is used later,
        // changing to `next_back()` may change its behavior.
        if !(is_mutable(cx, self_expr) || self_type.is_ref()) {
            if let Some(hir_id) = path_to_local(self_expr)
                && let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
                && let PatKind::Binding(_, _, ident, _) = pat.kind
            {
                sugg.push((ident.span.shrink_to_lo(), String::from("mut ")));
            } else {
                // If we can't make the binding mutable, make the suggestion `Unspecified` to prevent it from being
                // automatically applied, and add a complementary help message.
                dont_apply = true;
            }
        }
        span_lint_and_then(
            cx,
            DOUBLE_ENDED_ITERATOR_LAST,
            expr.span,
            "called `Iterator::last` on a `DoubleEndedIterator`; this will needlessly iterate the entire iterator",
            |diag| {
                let expr_ty = cx.typeck_results().expr_ty(expr);
                let droppable_elements = expr_ty.has_significant_drop(cx.tcx, cx.typing_env());
                diag.multipart_suggestion(
                    "try",
                    sugg,
                    if dont_apply {
                        Applicability::Unspecified
                    } else if droppable_elements {
                        Applicability::MaybeIncorrect
                    } else {
                        Applicability::MachineApplicable
                    },
                );
                if droppable_elements {
                    diag.note("this change will alter drop order which may be undesirable");
                }
                if dont_apply {
                    diag.span_note(self_expr.span, "this must be made mutable to use `.next_back()`");
                }
            },
        );
    }
}
