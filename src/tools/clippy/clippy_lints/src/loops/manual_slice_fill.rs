use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{HasSession, snippet_with_applicability};
use clippy_utils::ty::{implements_trait, is_slice_like};
use clippy_utils::visitors::is_local_used;
use clippy_utils::{higher, peel_blocks_with_stmt, span_contains_comment};
use rustc_ast::ast::LitKind;
use rustc_ast::{RangeLimits, UnOp};
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::QPath::Resolved;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, Pat};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use rustc_span::sym;

use super::MANUAL_SLICE_FILL;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    msrv: Msrv,
) {
    // `for _ in 0..slice.len() { slice[_] = value; }`
    if let Some(higher::Range {
        start: Some(start),
        end: Some(end),
        limits: RangeLimits::HalfOpen,
    }) = higher::Range::hir(arg)
        && let ExprKind::Lit(Spanned {
            node: LitKind::Int(Pu128(0), _),
            ..
        }) = start.kind
        && let ExprKind::Block(..) = body.kind
        // Check if the body is an assignment to a slice element.
        && let ExprKind::Assign(assignee, assignval, _) = peel_blocks_with_stmt(body).kind
        && let ExprKind::Index(slice, idx, _) = assignee.kind
        // Check if `len()` is used for the range end.
        && let ExprKind::MethodCall(path, recv,..) = end.kind
        && path.ident.name == sym::len
        // Check if the slice which is being assigned to is the same as the one being iterated over.
        && let ExprKind::Path(Resolved(_, recv_path)) = recv.kind
        && let ExprKind::Path(Resolved(_, slice_path)) = slice.kind
        && recv_path.res == slice_path.res
        && !assignval.span.from_expansion()
        // It is generally not equivalent to use the `fill` method if `assignval` can have side effects
        && switch_to_eager_eval(cx, assignval)
        // The `fill` method requires that the slice's element type implements the `Clone` trait.
        && let Some(clone_trait) = cx.tcx.lang_items().clone_trait()
        && implements_trait(cx, cx.typeck_results().expr_ty(slice), clone_trait, &[])
        // https://github.com/rust-lang/rust-clippy/issues/14192
        && let ExprKind::Path(Resolved(_, idx_path)) = idx.kind
        && let Res::Local(idx_hir) = idx_path.res
        && !is_local_used(cx, assignval, idx_hir)
        && msrv.meets(cx, msrvs::SLICE_FILL)
        && let slice_ty = cx.typeck_results().expr_ty(slice).peel_refs()
        && is_slice_like(cx, slice_ty)
    {
        sugg(cx, body, expr, slice.span, assignval.span);
    }
    // `for _ in &mut slice { *_ = value; }`
    else if let ExprKind::AddrOf(_, _, recv) = arg.kind
        // Check if the body is an assignment to a slice element.
        && let ExprKind::Assign(assignee, assignval, _) = peel_blocks_with_stmt(body).kind
        && let ExprKind::Unary(UnOp::Deref, slice_iter) = assignee.kind
        && let ExprKind::Path(Resolved(_, recv_path)) = recv.kind
        // Check if the slice which is being assigned to is the same as the one being iterated over.
        && let ExprKind::Path(Resolved(_, slice_path)) = slice_iter.kind
        && let Res::Local(local) = slice_path.res
        && local == pat.hir_id
        && !assignval.span.from_expansion()
        && switch_to_eager_eval(cx, assignval)
        // `assignval` must not reference the iterator
        && !is_local_used(cx, assignval, local)
        // The `fill` method cannot be used if the slice's element type does not implement the `Clone` trait.
        && let Some(clone_trait) = cx.tcx.lang_items().clone_trait()
        && implements_trait(cx, cx.typeck_results().expr_ty(recv), clone_trait, &[])
        && msrv.meets(cx, msrvs::SLICE_FILL)
    {
        sugg(cx, body, expr, recv_path.span, assignval.span);
    }
}

fn sugg<'tcx>(
    cx: &LateContext<'tcx>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    slice_span: rustc_span::Span,
    assignval_span: rustc_span::Span,
) {
    let mut app = if span_contains_comment(cx.sess().source_map(), body.span) {
        Applicability::MaybeIncorrect // Comments may be informational.
    } else {
        Applicability::MachineApplicable
    };

    span_lint_and_sugg(
        cx,
        MANUAL_SLICE_FILL,
        expr.span,
        "manually filling a slice",
        "try",
        format!(
            "{}.fill({});",
            snippet_with_applicability(cx, slice_span, "..", &mut app),
            snippet_with_applicability(cx, assignval_span, "..", &mut app),
        ),
        app,
    );
}
