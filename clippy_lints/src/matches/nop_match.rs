#![allow(unused_variables)]
use super::NOP_MATCH;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{eq_expr_value, get_parent_expr};
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingAnnotation, Expr, ExprKind, Pat, PatKind, PathSegment, QPath};
use rustc_lint::LateContext;

pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>) {
    if false {
        span_lint_and_sugg(
            cx,
            NOP_MATCH,
            ex.span,
            "this if-let expression is unnecessary",
            "replace it with",
            "".to_string(),
            Applicability::MachineApplicable,
        );
    }
}

pub(crate) fn check_match(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    // This is for avoiding collision with `match_single_binding`.
    if arms.len() < 2 {
        return;
    }

    for arm in arms {
        if let PatKind::Wild = arm.pat.kind {
            let ret_expr = strip_return(arm.body);
            if !eq_expr_value(cx, ex, ret_expr) {
                return;
            }
        } else if !pat_same_as_expr(arm.pat, arm.body) {
            return;
        }
    }

    if let Some(match_expr) = get_parent_expr(cx, ex) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            NOP_MATCH,
            match_expr.span,
            "this match expression is unnecessary",
            "replace it with",
            snippet_with_applicability(cx, ex.span, "..", &mut applicability).to_string(),
            applicability,
        );
    }
}

fn strip_return<'hir>(expr: &'hir Expr<'hir>) -> &'hir Expr<'hir> {
    if let ExprKind::Ret(Some(ret)) = expr.kind {
        ret
    } else {
        expr
    }
}

fn pat_same_as_expr(pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
    let expr = strip_return(expr);
    match (&pat.kind, &expr.kind) {
        (
            PatKind::TupleStruct(QPath::Resolved(_, path), [first_pat, ..], _),
            ExprKind::Call(call_expr, [first_param, ..]),
        ) => {
            if let ExprKind::Path(QPath::Resolved(_, call_path)) = call_expr.kind {
                if is_identical_segments(path.segments, call_path.segments)
                    && has_same_non_ref_symbol(first_pat, first_param)
                {
                    return true;
                }
            }
        },
        (PatKind::Path(QPath::Resolved(_, p_path)), ExprKind::Path(QPath::Resolved(_, e_path))) => {
            return is_identical_segments(p_path.segments, e_path.segments);
        },
        (PatKind::Lit(pat_lit_expr), ExprKind::Lit(expr_spanned)) => {
            if let ExprKind::Lit(pat_spanned) = &pat_lit_expr.kind {
                return pat_spanned.node == expr_spanned.node;
            }
        },
        _ => {},
    }

    false
}

fn is_identical_segments(left_segs: &[PathSegment<'_>], right_segs: &[PathSegment<'_>]) -> bool {
    if left_segs.len() != right_segs.len() {
        return false;
    }
    for i in 0..left_segs.len() {
        if left_segs[i].ident.name != right_segs[i].ident.name {
            return false;
        }
    }
    true
}

fn has_same_non_ref_symbol(pat: &Pat<'_>, expr: &Expr<'_>) -> bool {
    if_chain! {
        if let PatKind::Binding(annot, _, pat_ident, _) = pat.kind;
        if !matches!(annot, BindingAnnotation::Ref | BindingAnnotation::RefMut);
        if let ExprKind::Path(QPath::Resolved(_, path)) = expr.kind;
        if let Some(first_seg) = path.segments.first();
        then {
            return pat_ident.name == first_seg.ident.name;
        }
    }

    false
}
