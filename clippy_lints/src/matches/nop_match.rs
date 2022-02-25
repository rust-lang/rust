#![allow(unused_variables)]
use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_lint::LateContext;
use rustc_hir::{Arm, Expr};
use rustc_errors::Applicability;
use super::NOP_MATCH;

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
    if false {
        span_lint_and_sugg(
            cx,
            NOP_MATCH,
            ex.span,
            "this match expression is unnecessary",
            "replace it with",
            "".to_string(),
            Applicability::MachineApplicable,
        );
    }
}