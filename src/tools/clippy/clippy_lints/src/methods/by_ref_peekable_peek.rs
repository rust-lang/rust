use crate::clippy_utils::res::MaybeTypeckRes;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeResPath as _};
use clippy_utils::sugg::Sugg;
use clippy_utils::sym;
use clippy_utils::ty::implements_trait;
use rustc_ast::BindingMode;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LetStmt, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::BY_REF_PEEKABLE_PEEK;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>) {
    if let ExprKind::MethodCall(maybe_peekable, peekable_recv, [], _) = recv.kind
        && maybe_peekable.ident.name == sym::peekable
        && !peekable_recv.span.from_expansion()
        && let ExprKind::MethodCall(maybe_by_ref, by_ref_recv, [], _) = peekable_recv.kind
        && maybe_by_ref.ident.name == sym::by_ref
        && !by_ref_recv.span.from_expansion()
        && [peekable_recv, recv]
            .into_iter()
            .all(|e| cx.ty_based_def(e).opt_parent(cx).is_diag_item(cx, sym::Iterator))
    {
        span_lint_and_then(
            cx,
            BY_REF_PEEKABLE_PEEK,
            expr.span,
            "calling `.by_ref().peekable().peek()` will advance the underlying iterator and consume its first output",
            |diag| {
                let span = by_ref_recv.span.shrink_to_hi().with_hi(expr.span.hi());
                if let ty::Ref(_, iter_ty, _) = cx.typeck_results().expr_ty_adjusted(by_ref_recv).kind()
                    && let Some(clone_trait) = cx.tcx.lang_items().clone_trait()
                    && implements_trait(cx, *iter_ty, clone_trait, &[])
                {
                    diag.span_suggestion_verbose(
                        span,
                        "to peek the first item without advancing the underlying iterator, use",
                        ".clone().next().as_ref()",
                        Applicability::MaybeIncorrect,
                    );
                }
                diag.span_suggestion_verbose(
                    span,
                    "to advance the underlying iterator, use",
                    ".next().as_ref()",
                    Applicability::MaybeIncorrect,
                );
                // If the iterator is a local variable, initialized through a simple binding with an inferred
                // initialization expression, suggest making the initialization expression peekable.
                if let Some(iter_local_id) = by_ref_recv.res_local_id()
                    && let Node::LetStmt(LetStmt {
                        pat: let_pat,
                        ty: None,
                        init: Some(init_expr),
                        els: None,
                        span: let_stmt_span,
                        ..
                    }) = cx.tcx.parent_hir_node(iter_local_id)
                    && let PatKind::Binding(BindingMode::MUT, _, _, None) = let_pat.kind
                    && !let_stmt_span.from_expansion()
                    // Changing the type of the iterator may prevent the code from compiling
                    && let mut app = Applicability::MaybeIncorrect
                    && let sugg =
                        Sugg::hir_with_context(cx, init_expr, let_stmt_span.ctxt(), "_", &mut app).maybe_paren()
                {
                    diag.multipart_suggestion(
                        "to make the iterator peekable, use",
                        vec![
                            (init_expr.span.source_callsite(), format!("{sugg}.peekable()")),
                            (recv.span.with_lo(by_ref_recv.span.hi()), String::new()),
                        ],
                        app,
                    );
                } else {
                    diag.help(
                        "you might want to transform the iterator itself using `.peekable()` without using `.by_ref()`",
                    );
                }
            },
        );
    }
}
