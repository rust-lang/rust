use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::msrvs::Msrv;
use clippy_utils::source::snippet;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{
    SpanlessEq, get_ref_operators, is_res_lang_ctor, is_unit_expr, path_to_local, peel_blocks_with_stmt,
    peel_ref_operators,
};
use rustc_ast::BorrowKind;
use rustc_errors::MultiSpan;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{Arm, Expr, ExprKind, HirId, Pat, PatExpr, PatExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::{COLLAPSIBLE_MATCH, pat_contains_disallowed_or};

pub(super) fn check_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>], msrv: Msrv) {
    if let Some(els_arm) = arms.iter().rfind(|arm| arm_is_wild_like(cx, arm)) {
        for arm in arms {
            check_arm(cx, true, arm.pat, expr, arm.body, arm.guard, Some(els_arm.body), msrv);
        }
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    body: &'tcx Expr<'_>,
    else_expr: Option<&'tcx Expr<'_>>,
    let_expr: &'tcx Expr<'_>,
    msrv: Msrv,
) {
    check_arm(cx, false, pat, let_expr, body, None, else_expr, msrv);
}

#[allow(clippy::too_many_arguments)]
fn check_arm<'tcx>(
    cx: &LateContext<'tcx>,
    outer_is_match: bool,
    outer_pat: &'tcx Pat<'tcx>,
    outer_cond: &'tcx Expr<'tcx>,
    outer_then_body: &'tcx Expr<'tcx>,
    outer_guard: Option<&'tcx Expr<'tcx>>,
    outer_else_body: Option<&'tcx Expr<'tcx>>,
    msrv: Msrv,
) {
    let inner_expr = peel_blocks_with_stmt(outer_then_body);
    if let Some(inner) = IfLetOrMatch::parse(cx, inner_expr)
        && let Some((inner_scrutinee, inner_then_pat, inner_else_body)) = match inner {
            IfLetOrMatch::IfLet(scrutinee, pat, _, els, _) => Some((scrutinee, pat, els)),
            IfLetOrMatch::Match(scrutinee, arms, ..) => if arms.len() == 2 && arms.iter().all(|a| a.guard.is_none())
                // if there are more than two arms, collapsing would be non-trivial
                // one of the arms must be "wild-like"
                && let Some(wild_idx) = arms.iter().rposition(|a| arm_is_wild_like(cx, a))
            {
                let (then, els) = (&arms[1 - wild_idx], &arms[wild_idx]);
                Some((scrutinee, then.pat, Some(els.body)))
            } else {
                None
            },
        }
        && outer_pat.span.eq_ctxt(inner_scrutinee.span)
        // match expression must be a local binding
        // match <local> { .. }
        && let Some(binding_id) = path_to_local(peel_ref_operators(cx, inner_scrutinee))
        && !pat_contains_disallowed_or(cx, inner_then_pat, msrv)
        // the binding must come from the pattern of the containing match arm
        // ..<local>.. => match <local> { .. }
        && let (Some(binding_span), is_innermost_parent_pat_struct)
            = find_pat_binding_and_is_innermost_parent_pat_struct(outer_pat, binding_id)
        // the "else" branches must be equal
        && match (outer_else_body, inner_else_body) {
            (None, None) => true,
            (None, Some(e)) | (Some(e), None) => is_unit_expr(e),
            (Some(a), Some(b)) => SpanlessEq::new(cx).eq_expr(a, b),
        }
        // the binding must not be used in the if guard
        && outer_guard.is_none_or(
            |e| !is_local_used(cx, e, binding_id)
        )
        // ...or anywhere in the inner expression
        && match inner {
            IfLetOrMatch::IfLet(_, _, body, els, _) => {
                !is_local_used(cx, body, binding_id) && els.is_none_or(|e| !is_local_used(cx, e, binding_id))
            },
            IfLetOrMatch::Match(_, arms, ..) => !arms.iter().any(|arm| is_local_used(cx, arm, binding_id)),
        }
        // Check if the inner expression contains any borrows/dereferences
        && let ref_types = get_ref_operators(cx, inner_scrutinee)
        && let Some(method) = build_ref_method_chain(ref_types)
    {
        let msg = format!(
            "this `{}` can be collapsed into the outer `{}`",
            if matches!(inner, IfLetOrMatch::Match(..)) {
                "match"
            } else {
                "if let"
            },
            if outer_is_match { "match" } else { "if let" },
        );
        // collapsing patterns need an explicit field name in struct pattern matching
        // ex: Struct {x: Some(1)}
        let replace_msg = if is_innermost_parent_pat_struct {
            format!(", prefixed by `{}`:", snippet(cx, binding_span, "their field name"))
        } else {
            String::new()
        };
        span_lint_hir_and_then(cx, COLLAPSIBLE_MATCH, inner_expr.hir_id, inner_expr.span, msg, |diag| {
            let mut help_span = MultiSpan::from_spans(vec![binding_span, inner_then_pat.span]);
            help_span.push_span_label(binding_span, "replace this binding");
            help_span.push_span_label(inner_then_pat.span, format!("with this pattern{replace_msg}"));
            if !method.is_empty() {
                let outer_cond_msg = format!("use: `{}{}`", snippet(cx, outer_cond.span, ".."), method);
                help_span.push_span_label(outer_cond.span, outer_cond_msg);
            }
            diag.span_help(
                help_span,
                "the outer pattern can be modified to include the inner pattern",
            );
        });
    }
}

/// A "wild-like" arm has a wild (`_`) or `None` pattern and no guard. Such arms can be "collapsed"
/// into a single wild arm without any significant loss in semantics or readability.
fn arm_is_wild_like(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    if arm.guard.is_some() {
        return false;
    }
    match arm.pat.kind {
        PatKind::Binding(..) | PatKind::Wild => true,
        PatKind::Expr(PatExpr {
            kind: PatExprKind::Path(qpath),
            hir_id,
            ..
        }) => is_res_lang_ctor(cx, cx.qpath_res(qpath, *hir_id), OptionNone),
        _ => false,
    }
}

fn find_pat_binding_and_is_innermost_parent_pat_struct(pat: &Pat<'_>, hir_id: HirId) -> (Option<Span>, bool) {
    let mut span = None;
    let mut is_innermost_parent_pat_struct = false;
    pat.walk_short(|p| match &p.kind {
        // ignore OR patterns
        PatKind::Or(_) => false,
        PatKind::Binding(_bm, _, _ident, _) => {
            let found = p.hir_id == hir_id;
            if found {
                span = Some(p.span);
            }
            !found
        },
        _ => {
            is_innermost_parent_pat_struct = matches!(p.kind, PatKind::Struct(..));
            true
        },
    });
    (span, is_innermost_parent_pat_struct)
}

/// Builds a chain of reference-manipulation method calls (e.g., `.as_ref()`, `.as_mut()`,
/// `.copied()`) based on reference operators
fn build_ref_method_chain(expr: Vec<&Expr<'_>>) -> Option<String> {
    let mut req_method_calls = String::new();

    for ref_operator in expr {
        match ref_operator.kind {
            ExprKind::AddrOf(BorrowKind::Raw, _, _) => {
                return None;
            },
            ExprKind::AddrOf(_, m, _) if m.is_mut() => {
                req_method_calls.push_str(".as_mut()");
            },
            ExprKind::AddrOf(_, _, _) => {
                req_method_calls.push_str(".as_ref()");
            },
            // Deref operator is the only operator that this function should have received
            ExprKind::Unary(_, _) => {
                req_method_calls.push_str(".copied()");
            },
            _ => (),
        }
    }

    Some(req_method_calls)
}
