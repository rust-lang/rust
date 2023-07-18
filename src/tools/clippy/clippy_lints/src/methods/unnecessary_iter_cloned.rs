use super::utils::clone_or_copy_needed;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{get_iterator_item_ty, implements_trait};
use clippy_utils::{fn_def_id, get_parent_expr};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::{sym, Symbol};

use super::UNNECESSARY_TO_OWNED;

pub fn check(cx: &LateContext<'_>, expr: &Expr<'_>, method_name: Symbol, receiver: &Expr<'_>) -> bool {
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let Some(callee_def_id) = fn_def_id(cx, parent);
        if is_into_iter(cx, callee_def_id);
        then {
            check_for_loop_iter(cx, parent, method_name, receiver, false)
        } else {
            false
        }
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
    if_chain! {
        if let Some(grandparent) = get_parent_expr(cx, expr).and_then(|parent| get_parent_expr(cx, parent));
        if let Some(ForLoop { pat, body, .. }) = ForLoop::hir(grandparent);
        let (clone_or_copy_needed, addr_of_exprs) = clone_or_copy_needed(cx, pat, body);
        if !clone_or_copy_needed;
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            let snippet = if_chain! {
                if let ExprKind::MethodCall(maybe_iter_method_name, collection, [], _) = receiver.kind;
                if maybe_iter_method_name.ident.name == sym::iter;

                if let Some(iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
                let receiver_ty = cx.typeck_results().expr_ty(receiver);
                if implements_trait(cx, receiver_ty, iterator_trait_id, &[]);
                if let Some(iter_item_ty) = get_iterator_item_ty(cx, receiver_ty);

                if let Some(into_iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator);
                let collection_ty = cx.typeck_results().expr_ty(collection);
                if implements_trait(cx, collection_ty, into_iterator_trait_id, &[]);
                if let Some(into_iter_item_ty) = cx.get_associated_type(collection_ty, into_iterator_trait_id, "Item");

                if iter_item_ty == into_iter_item_ty;
                if let Some(collection_snippet) = snippet_opt(cx, collection.span);
                then {
                    collection_snippet
                } else {
                    receiver_snippet
                }
            };
            span_lint_and_then(
                cx,
                UNNECESSARY_TO_OWNED,
                expr.span,
                &format!("unnecessary use of `{method_name}`"),
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
                    diag.span_suggestion(expr.span, "use", snippet, applicability);
                    for addr_of_expr in addr_of_exprs {
                        match addr_of_expr.kind {
                            ExprKind::AddrOf(_, _, referent) => {
                                let span = addr_of_expr.span.with_hi(referent.span.lo());
                                diag.span_suggestion(span, "remove this `&`", "", applicability);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            );
            return true;
        }
    }
    false
}

/// Returns true if the named method is `IntoIterator::into_iter`.
pub fn is_into_iter(cx: &LateContext<'_>, callee_def_id: DefId) -> bool {
    Some(callee_def_id) == cx.tcx.lang_items().into_iter_fn()
}
