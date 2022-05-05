use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_integer_const;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{
    higher::{self, Range},
    SpanlessEq,
};
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;

use super::ITER_WITH_DRAIN;

const DRAIN_TYPES: &[Symbol] = &[sym::Vec, sym::VecDeque];

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: &Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();
    if let Some(drained_type) = DRAIN_TYPES.iter().find(|&&sym| is_type_diagnostic_item(cx, ty, sym)) {
        // Refuse to emit `into_iter` suggestion on draining struct fields due
        // to the strong possibility of processing unmovable field.
        if let ExprKind::Field(..) = recv.kind {
            return;
        }

        if let Some(range) = higher::Range::hir(arg) {
            let left_full = match range {
                Range { start: Some(start), .. } if is_integer_const(cx, start, 0) => true,
                Range { start: None, .. } => true,
                _ => false,
            };
            let full = left_full
                && match range {
                    Range {
                        end: Some(end),
                        limits: RangeLimits::HalfOpen,
                        ..
                    } => {
                        // `x.drain(..x.len())` call
                        if_chain! {
                            if let ExprKind::MethodCall(len_path, len_args, _) = end.kind;
                            if len_path.ident.name == sym::len && len_args.len() == 1;
                            if let ExprKind::Path(QPath::Resolved(_, drain_path)) = recv.kind;
                            if let ExprKind::Path(QPath::Resolved(_, len_path)) = len_args[0].kind;
                            if SpanlessEq::new(cx).eq_path(drain_path, len_path);
                            then { true }
                            else { false }
                        }
                    },
                    Range {
                        end: None,
                        limits: RangeLimits::HalfOpen,
                        ..
                    } => true,
                    _ => false,
                };
            if full {
                span_lint_and_sugg(
                    cx,
                    ITER_WITH_DRAIN,
                    span.with_hi(expr.span.hi()),
                    &format!("`drain(..)` used on a `{}`", drained_type),
                    "try this",
                    "into_iter()".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
