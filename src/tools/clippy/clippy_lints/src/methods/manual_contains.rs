use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::peel_hir_pat_refs;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_ast::UnOp;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, HirId, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self};
use rustc_span::source_map::Spanned;

use super::MANUAL_CONTAINS;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, closure_arg: &Expr<'_>) {
    let mut app = Applicability::MachineApplicable;

    if !expr.span.from_expansion()
        // check if `iter().any()` can be replaced with `contains()`
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let Body{params: [param],value} = cx.tcx.hir_body(closure.body)
        && let ExprKind::Binary(op, lhs, rhs) = value.kind
        && let (peeled_ref_pat, _) = peel_hir_pat_refs(param.pat)
        && let Some((snip,snip_expr)) = can_replace_with_contains(cx, op, lhs, rhs, peeled_ref_pat.hir_id, &mut app)
        && let ref_type = cx.typeck_results().expr_ty_adjusted(recv)
        && let ty::Ref(_, inner_type, _) = ref_type.kind()
        && let ty::Slice(slice_type) = inner_type.kind()
        && *slice_type == cx.typeck_results().expr_ty(snip_expr)
    {
        span_lint_and_sugg(
            cx,
            MANUAL_CONTAINS,
            expr.span,
            "using `contains()` instead of `iter().any()` is more efficient",
            "try",
            format!(
                "{}.contains({})",
                snippet_with_applicability(cx, recv.span, "_", &mut app),
                snip
            ),
            app,
        );
    }
}

enum EligibleArg {
    IsClosureArg,
    ContainsArg(String),
}

fn try_get_eligible_arg<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    closure_arg_id: HirId,
    applicability: &mut Applicability,
) -> Option<(EligibleArg, &'tcx Expr<'tcx>)> {
    let mut get_snippet = |expr: &Expr<'_>, needs_borrow: bool| {
        let sugg = Sugg::hir_with_applicability(cx, expr, "_", applicability);
        EligibleArg::ContainsArg((if needs_borrow { sugg.addr() } else { sugg }).to_string())
    };

    match expr.kind {
        ExprKind::Path(QPath::Resolved(_, path)) => {
            if path.res == Res::Local(closure_arg_id) {
                Some((EligibleArg::IsClosureArg, expr))
            } else {
                Some((get_snippet(expr, true), expr))
            }
        },
        ExprKind::Unary(UnOp::Deref, inner) => {
            if let ExprKind::Path(QPath::Resolved(_, path)) = inner.kind {
                if path.res == Res::Local(closure_arg_id) {
                    Some((EligibleArg::IsClosureArg, expr))
                } else {
                    Some((get_snippet(inner, false), expr))
                }
            } else {
                None
            }
        },
        _ => {
            if switch_to_eager_eval(cx, expr) {
                Some((get_snippet(expr, true), expr))
            } else {
                None
            }
        },
    }
}

fn can_replace_with_contains<'tcx>(
    cx: &LateContext<'tcx>,
    bin_op: Spanned<BinOpKind>,
    left_expr: &'tcx Expr<'tcx>,
    right_expr: &'tcx Expr<'tcx>,
    closure_arg_id: HirId,
    applicability: &mut Applicability,
) -> Option<(String, &'tcx Expr<'tcx>)> {
    if bin_op.node != BinOpKind::Eq {
        return None;
    }

    let left_candidate = try_get_eligible_arg(cx, left_expr, closure_arg_id, applicability)?;
    let right_candidate = try_get_eligible_arg(cx, right_expr, closure_arg_id, applicability)?;
    match (left_candidate, right_candidate) {
        ((EligibleArg::IsClosureArg, _), (EligibleArg::ContainsArg(snip), candidate_expr))
        | ((EligibleArg::ContainsArg(snip), candidate_expr), (EligibleArg::IsClosureArg, _)) => {
            Some((snip, candidate_expr))
        },
        _ => None,
    }
}
