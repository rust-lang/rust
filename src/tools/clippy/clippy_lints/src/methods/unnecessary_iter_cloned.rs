use super::utils::clone_or_copy_needed;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::{get_iterator_item_ty, implements_trait};
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{can_mut_borrow_both, fn_def_id, get_parent_expr, path_to_local};
use core::ops::ControlFlow;
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BindingMode, Expr, ExprKind, Node, PatKind};
use rustc_lint::LateContext;
use rustc_span::{Symbol, sym};

use super::UNNECESSARY_TO_OWNED;

pub fn check(cx: &LateContext<'_>, expr: &Expr<'_>, method_name: Symbol, receiver: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, expr)
        && let Some(callee_def_id) = fn_def_id(cx, parent)
        && is_into_iter(cx, callee_def_id)
    {
        check_for_loop_iter(cx, parent, method_name, receiver, false)
    } else {
        false
    }
}

/// Checks whether `expr` is an iterator in a `for` loop and, if so, determines whether the
/// iterated-over items could be iterated over by reference. The reason why `check` above does not
/// include this code directly is so that it can be called from
/// `unnecessary_into_owned::check_into_iter_call_arg`.
pub fn check_for_loop_iter(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    method_name: Symbol,
    receiver: &Expr<'_>,
    cloned_before_iter: bool,
) -> bool {
    if let Some(grandparent) = get_parent_expr(cx, expr).and_then(|parent| get_parent_expr(cx, parent))
        && let Some(ForLoop { pat, body, .. }) = ForLoop::hir(grandparent)
        && let (clone_or_copy_needed, references_to_binding) = clone_or_copy_needed(cx, pat, body)
        && !clone_or_copy_needed
        && let Some(receiver_snippet) = receiver.span.get_source_text(cx)
    {
        // Issue 12098
        // https://github.com/rust-lang/rust-clippy/issues/12098
        // if the assignee have `mut borrow` conflict with the iteratee
        // the lint should not execute, former didn't consider the mut case

        // check whether `expr` is mutable
        fn is_mutable(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            if let Some(hir_id) = path_to_local(expr)
                && let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
            {
                matches!(pat.kind, PatKind::Binding(BindingMode::MUT, ..))
            } else {
                true
            }
        }

        fn is_caller_or_fields_change(cx: &LateContext<'_>, body: &Expr<'_>, caller: &Expr<'_>) -> bool {
            let mut change = false;
            if let ExprKind::Block(block, ..) = body.kind {
                for_each_expr_without_closures(block, |e| {
                    match e.kind {
                        ExprKind::Assign(assignee, _, _) | ExprKind::AssignOp(_, assignee, _) => {
                            change |= !can_mut_borrow_both(cx, caller, assignee);
                        },
                        _ => {},
                    }
                    // the return value has no effect but the function need one return value
                    ControlFlow::<()>::Continue(())
                });
            }
            change
        }

        if let ExprKind::Call(_, [child, ..]) = expr.kind {
            // filter first layer of iterator
            let mut child = child;
            // get inner real caller requests for clone
            while let ExprKind::MethodCall(_, caller, _, _) = child.kind {
                child = caller;
            }
            if is_mutable(cx, child) && is_caller_or_fields_change(cx, body, child) {
                // skip lint
                return true;
            }
        }

        // the lint should not be executed if no violation happens
        let snippet = if let ExprKind::MethodCall(maybe_iter_method_name, collection, [], _) = receiver.kind
            && maybe_iter_method_name.ident.name == sym::iter
            && let Some(iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
            && let receiver_ty = cx.typeck_results().expr_ty(receiver)
            && implements_trait(cx, receiver_ty, iterator_trait_id, &[])
            && let Some(iter_item_ty) = get_iterator_item_ty(cx, receiver_ty)
            && let Some(into_iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
            && let collection_ty = cx.typeck_results().expr_ty(collection)
            && implements_trait(cx, collection_ty, into_iterator_trait_id, &[])
            && let Some(into_iter_item_ty) = cx.get_associated_type(collection_ty, into_iterator_trait_id, sym::Item)
            && iter_item_ty == into_iter_item_ty
            && let Some(collection_snippet) = collection.span.get_source_text(cx)
        {
            collection_snippet
        } else {
            receiver_snippet
        };
        span_lint_and_then(
            cx,
            UNNECESSARY_TO_OWNED,
            expr.span,
            format!("unnecessary use of `{method_name}`"),
            |diag| {
                // If `check_into_iter_call_arg` called `check_for_loop_iter` because a call to
                // a `to_owned`-like function was removed, then the next suggestion may be
                // incorrect. This is because the iterator that results from the call's removal
                // could hold a reference to a resource that is used mutably. See
                // https://github.com/rust-lang/rust-clippy/issues/8148.
                let applicability = if cloned_before_iter {
                    Applicability::MaybeIncorrect
                } else {
                    Applicability::MachineApplicable
                };

                let combined = references_to_binding
                    .into_iter()
                    .chain(vec![(expr.span, snippet.to_owned())])
                    .collect_vec();

                diag.multipart_suggestion("remove any references to the binding", combined, applicability);
            },
        );
        return true;
    }
    false
}

/// Returns true if the named method is `IntoIterator::into_iter`.
pub fn is_into_iter(cx: &LateContext<'_>, callee_def_id: DefId) -> bool {
    Some(callee_def_id) == cx.tcx.lang_items().into_iter_fn()
}
