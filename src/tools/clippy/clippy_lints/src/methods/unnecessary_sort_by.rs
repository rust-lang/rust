use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_trait_method, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Closure, Expr, ExprKind, Mutability, Param, Pat, PatKind, Path, PathSegment, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArgKind};
use rustc_span::sym;
use rustc_span::symbol::Ident;
use std::iter;

use super::UNNECESSARY_SORT_BY;

enum LintTrigger {
    Sort(SortDetection),
    SortByKey(SortByKeyDetection),
}

struct SortDetection {
    vec_name: String,
}

struct SortByKeyDetection {
    vec_name: String,
    closure_arg: String,
    closure_body: String,
    reverse: bool,
}

/// Detect if the two expressions are mirrored (identical, except one
/// contains a and the other replaces it with b)
fn mirrored_exprs(a_expr: &Expr<'_>, a_ident: &Ident, b_expr: &Expr<'_>, b_ident: &Ident) -> bool {
    match (&a_expr.kind, &b_expr.kind) {
        // Two arrays with mirrored contents
        (ExprKind::Array(left_exprs), ExprKind::Array(right_exprs)) => {
            iter::zip(*left_exprs, *right_exprs).all(|(left, right)| mirrored_exprs(left, a_ident, right, b_ident))
        },
        // The two exprs are function calls.
        // Check to see that the function itself and its arguments are mirrored
        (ExprKind::Call(left_expr, left_args), ExprKind::Call(right_expr, right_args)) => {
            mirrored_exprs(left_expr, a_ident, right_expr, b_ident)
                && iter::zip(*left_args, *right_args).all(|(left, right)| mirrored_exprs(left, a_ident, right, b_ident))
        },
        // The two exprs are method calls.
        // Check to see that the function is the same and the arguments and receivers are mirrored
        (
            ExprKind::MethodCall(left_segment, left_receiver, left_args, _),
            ExprKind::MethodCall(right_segment, right_receiver, right_args, _),
        ) => {
            left_segment.ident == right_segment.ident
                && iter::zip(*left_args, *right_args).all(|(left, right)| mirrored_exprs(left, a_ident, right, b_ident))
                && mirrored_exprs(left_receiver, a_ident, right_receiver, b_ident)
        },
        // Two tuples with mirrored contents
        (ExprKind::Tup(left_exprs), ExprKind::Tup(right_exprs)) => {
            iter::zip(*left_exprs, *right_exprs).all(|(left, right)| mirrored_exprs(left, a_ident, right, b_ident))
        },
        // Two binary ops, which are the same operation and which have mirrored arguments
        (ExprKind::Binary(left_op, left_left, left_right), ExprKind::Binary(right_op, right_left, right_right)) => {
            left_op.node == right_op.node
                && mirrored_exprs(left_left, a_ident, right_left, b_ident)
                && mirrored_exprs(left_right, a_ident, right_right, b_ident)
        },
        // Two unary ops, which are the same operation and which have the same argument
        (ExprKind::Unary(left_op, left_expr), ExprKind::Unary(right_op, right_expr)) => {
            left_op == right_op && mirrored_exprs(left_expr, a_ident, right_expr, b_ident)
        },
        // The two exprs are literals of some kind
        (ExprKind::Lit(left_lit), ExprKind::Lit(right_lit)) => left_lit.node == right_lit.node,
        (ExprKind::Cast(left, _), ExprKind::Cast(right, _)) => mirrored_exprs(left, a_ident, right, b_ident),
        (ExprKind::DropTemps(left_block), ExprKind::DropTemps(right_block)) => {
            mirrored_exprs(left_block, a_ident, right_block, b_ident)
        },
        (ExprKind::Field(left_expr, left_ident), ExprKind::Field(right_expr, right_ident)) => {
            left_ident.name == right_ident.name && mirrored_exprs(left_expr, a_ident, right_expr, right_ident)
        },
        // Two paths: either one is a and the other is b, or they're identical to each other
        (
            ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: left_segments,
                    ..
                },
            )),
            ExprKind::Path(QPath::Resolved(
                _,
                Path {
                    segments: right_segments,
                    ..
                },
            )),
        ) => {
            (iter::zip(*left_segments, *right_segments).all(|(left, right)| left.ident == right.ident)
                && left_segments
                    .iter()
                    .all(|seg| &seg.ident != a_ident && &seg.ident != b_ident))
                || (left_segments.len() == 1
                    && &left_segments[0].ident == a_ident
                    && right_segments.len() == 1
                    && &right_segments[0].ident == b_ident)
        },
        // Matching expressions, but one or both is borrowed
        (
            ExprKind::AddrOf(left_kind, Mutability::Not, left_expr),
            ExprKind::AddrOf(right_kind, Mutability::Not, right_expr),
        ) => left_kind == right_kind && mirrored_exprs(left_expr, a_ident, right_expr, b_ident),
        (_, ExprKind::AddrOf(_, Mutability::Not, right_expr)) => mirrored_exprs(a_expr, a_ident, right_expr, b_ident),
        (ExprKind::AddrOf(_, Mutability::Not, left_expr), _) => mirrored_exprs(left_expr, a_ident, b_expr, b_ident),
        _ => false,
    }
}

fn detect_lint(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) -> Option<LintTrigger> {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && cx.tcx.type_of(impl_id).instantiate_identity().is_slice()
        && let ExprKind::Closure(&Closure { body, .. }) = arg.kind
        && let closure_body = cx.tcx.hir_body(body)
        && let &[
            Param {
                pat:
                    Pat {
                        kind: PatKind::Binding(_, _, left_ident, _),
                        ..
                    },
                ..
            },
            Param {
                pat:
                    Pat {
                        kind: PatKind::Binding(_, _, right_ident, _),
                        ..
                    },
                ..
            },
        ] = &closure_body.params
        && let ExprKind::MethodCall(method_path, left_expr, [right_expr], _) = closure_body.value.kind
        && method_path.ident.name == sym::cmp
        && is_trait_method(cx, closure_body.value, sym::Ord)
    {
        let (closure_body, closure_arg, reverse) = if mirrored_exprs(left_expr, left_ident, right_expr, right_ident) {
            (
                Sugg::hir(cx, left_expr, "..").to_string(),
                left_ident.name.to_string(),
                false,
            )
        } else if mirrored_exprs(left_expr, right_ident, right_expr, left_ident) {
            (
                Sugg::hir(cx, left_expr, "..").to_string(),
                right_ident.name.to_string(),
                true,
            )
        } else {
            return None;
        };
        let vec_name = Sugg::hir(cx, recv, "..").to_string();

        if let ExprKind::Path(QPath::Resolved(
            _,
            Path {
                segments: [PathSegment { ident: left_name, .. }],
                ..
            },
        )) = &left_expr.kind
            && left_name == left_ident
            && cx
                .tcx
                .get_diagnostic_item(sym::Ord)
                .is_some_and(|id| implements_trait(cx, cx.typeck_results().expr_ty(left_expr), id, &[]))
        {
            return Some(LintTrigger::Sort(SortDetection { vec_name }));
        }

        if !expr_borrows(cx, left_expr) {
            return Some(LintTrigger::SortByKey(SortByKeyDetection {
                vec_name,
                closure_arg,
                closure_body,
                reverse,
            }));
        }
    }

    None
}

fn expr_borrows(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    matches!(ty.kind(), ty::Ref(..)) || ty.walk().any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(_)))
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
    is_unstable: bool,
) {
    match detect_lint(cx, expr, recv, arg) {
        Some(LintTrigger::SortByKey(trigger)) => span_lint_and_sugg(
            cx,
            UNNECESSARY_SORT_BY,
            expr.span,
            "consider using `sort_by_key`",
            "try",
            format!(
                "{}.sort{}_by_key(|{}| {})",
                trigger.vec_name,
                if is_unstable { "_unstable" } else { "" },
                trigger.closure_arg,
                if let Some(std_or_core) = std_or_core(cx)
                    && trigger.reverse
                {
                    format!("{}::cmp::Reverse({})", std_or_core, trigger.closure_body)
                } else {
                    trigger.closure_body.to_string()
                },
            ),
            if trigger.reverse {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            },
        ),
        Some(LintTrigger::Sort(trigger)) => span_lint_and_sugg(
            cx,
            UNNECESSARY_SORT_BY,
            expr.span,
            "consider using `sort`",
            "try",
            format!(
                "{}.sort{}()",
                trigger.vec_name,
                if is_unstable { "_unstable" } else { "" },
            ),
            Applicability::MachineApplicable,
        ),
        None => {},
    }
}
