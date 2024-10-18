use super::READONLY_WRITE_LOCK;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::mir::{enclosing_mir, visit_local_usage};
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::mir::{Location, START_BLOCK};
use rustc_span::sym;

fn is_unwrap_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::MethodCall(path, receiver, [], _) = expr.kind
        && path.ident.name == sym::unwrap
    {
        is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(receiver).peel_refs(), sym::Result)
    } else {
        false
    }
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, receiver: &Expr<'_>) {
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(receiver).peel_refs(), sym::RwLock)
        && let Node::Expr(unwrap_call_expr) = cx.tcx.parent_hir_node(expr.hir_id)
        && is_unwrap_call(cx, unwrap_call_expr)
        && let parent = cx.tcx.parent_hir_node(unwrap_call_expr.hir_id)
        && let Node::LetStmt(local) = parent
        && let PatKind::Binding(.., ident, _) = local.pat.kind
        // if the binding is prefixed with `_`, it typically means
        // that this guard only exists to protect a section of code
        // rather than the contained data
        && !ident.as_str().starts_with('_')
        && let Some(mir) = enclosing_mir(cx.tcx, expr.hir_id)
        && let Some((local, _)) = mir
            .local_decls
            .iter_enumerated()
            .find(|(_, decl)| local.span.contains(decl.source_info.span))
        && let Some(usages) = visit_local_usage(
            &[local],
            mir,
            Location {
                block: START_BLOCK,
                statement_index: 0,
            },
        )
        && let [usage] = usages.as_slice()
    {
        let writer_never_mutated = usage.local_consume_or_mutate_locs.is_empty();

        if writer_never_mutated {
            span_lint_and_sugg(
                cx,
                READONLY_WRITE_LOCK,
                expr.span,
                "this write lock is used only for reading",
                "consider using a read lock instead",
                format!("{}.read()", snippet(cx, receiver.span, "<receiver>")),
                Applicability::MaybeIncorrect, // write lock might be intentional for enforcing exclusiveness
            );
        }
    }
}
