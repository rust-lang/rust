use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Symbol;
use std::fmt;

use super::PTR_OFFSET_WITH_CAST;

pub(super) fn check(
    cx: &LateContext<'_>,
    method: Symbol,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    arg: &Expr<'_>,
    msrv: Msrv,
) {
    // `pointer::add` and `pointer::wrapping_add` are only stable since 1.26.0. These functions
    // became const-stable in 1.61.0, the same version that `pointer::offset` became const-stable.
    if !msrv.meets(cx, msrvs::POINTER_ADD_SUB_METHODS) {
        return;
    }

    let method = match method {
        sym::offset => Method::Offset,
        sym::wrapping_offset => Method::WrappingOffset,
        _ => return,
    };

    if !cx.typeck_results().expr_ty_adjusted(recv).is_raw_ptr() {
        return;
    }

    // Check if the argument to the method call is a cast from usize.
    let cast_lhs_expr = match arg.kind {
        ExprKind::Cast(lhs, _) if cx.typeck_results().expr_ty(lhs).is_usize() => lhs,
        _ => return,
    };

    let ExprKind::MethodCall(method_name, _, _, _) = expr.kind else {
        return;
    };

    let msg = format!("use of `{method}` with a `usize` casted to an `isize`");
    span_lint_and_then(cx, PTR_OFFSET_WITH_CAST, expr.span, msg, |diag| {
        diag.multipart_suggestion(
            format!("use `{}` instead", method.suggestion()),
            vec![
                (method_name.ident.span, method.suggestion().to_string()),
                (arg.span.with_lo(cast_lhs_expr.span.hi()), String::new()),
            ],
            Applicability::MachineApplicable,
        );
    });
}

#[derive(Copy, Clone)]
enum Method {
    Offset,
    WrappingOffset,
}

impl Method {
    #[must_use]
    fn suggestion(self) -> &'static str {
        match self {
            Self::Offset => "add",
            Self::WrappingOffset => "wrapping_add",
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offset => write!(f, "offset"),
            Self::WrappingOffset => write!(f, "wrapping_offset"),
        }
    }
}
