use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::source::snippet_opt;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use std::fmt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the `offset` pointer method with a `usize` casted to an
    /// `isize`.
    ///
    /// ### Why is this bad?
    /// If we’re always increasing the pointer address, we can avoid the numeric
    /// cast by using the `add` method instead.
    ///
    /// ### Example
    /// ```rust
    /// let vec = vec![b'a', b'b', b'c'];
    /// let ptr = vec.as_ptr();
    /// let offset = 1_usize;
    ///
    /// unsafe {
    ///     ptr.offset(offset as isize);
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// let vec = vec![b'a', b'b', b'c'];
    /// let ptr = vec.as_ptr();
    /// let offset = 1_usize;
    ///
    /// unsafe {
    ///     ptr.add(offset);
    /// }
    /// ```
    #[clippy::version = "1.30.0"]
    pub PTR_OFFSET_WITH_CAST,
    complexity,
    "unneeded pointer offset cast"
}

declare_lint_pass!(PtrOffsetWithCast => [PTR_OFFSET_WITH_CAST]);

impl<'tcx> LateLintPass<'tcx> for PtrOffsetWithCast {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Check if the expressions is a ptr.offset or ptr.wrapping_offset method call
        let Some((receiver_expr, arg_expr, method)) = expr_as_ptr_offset_call(cx, expr) else {
            return;
        };

        // Check if the argument to the method call is a cast from usize
        let Some(cast_lhs_expr) = expr_as_cast_from_usize(cx, arg_expr) else {
            return;
        };

        let msg = format!("use of `{method}` with a `usize` casted to an `isize`");
        if let Some(sugg) = build_suggestion(cx, method, receiver_expr, cast_lhs_expr) {
            span_lint_and_sugg(
                cx,
                PTR_OFFSET_WITH_CAST,
                expr.span,
                &msg,
                "try",
                sugg,
                Applicability::MachineApplicable,
            );
        } else {
            span_lint(cx, PTR_OFFSET_WITH_CAST, expr.span, &msg);
        }
    }
}

// If the given expression is a cast from a usize, return the lhs of the cast
fn expr_as_cast_from_usize<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Cast(cast_lhs_expr, _) = expr.kind {
        if is_expr_ty_usize(cx, cast_lhs_expr) {
            return Some(cast_lhs_expr);
        }
    }
    None
}

// If the given expression is a ptr::offset  or ptr::wrapping_offset method call, return the
// receiver, the arg of the method call, and the method.
fn expr_as_ptr_offset_call<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, Method)> {
    if let ExprKind::MethodCall(path_segment, arg_0, [arg_1, ..], _) = &expr.kind {
        if is_expr_ty_raw_ptr(cx, arg_0) {
            if path_segment.ident.name == sym::offset {
                return Some((arg_0, arg_1, Method::Offset));
            }
            if path_segment.ident.name == sym!(wrapping_offset) {
                return Some((arg_0, arg_1, Method::WrappingOffset));
            }
        }
    }
    None
}

// Is the type of the expression a usize?
fn is_expr_ty_usize(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    cx.typeck_results().expr_ty(expr) == cx.tcx.types.usize
}

// Is the type of the expression a raw pointer?
fn is_expr_ty_raw_ptr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    cx.typeck_results().expr_ty(expr).is_unsafe_ptr()
}

fn build_suggestion(
    cx: &LateContext<'_>,
    method: Method,
    receiver_expr: &Expr<'_>,
    cast_lhs_expr: &Expr<'_>,
) -> Option<String> {
    let receiver = snippet_opt(cx, receiver_expr.span)?;
    let cast_lhs = snippet_opt(cx, cast_lhs_expr.span)?;
    Some(format!("{receiver}.{}({cast_lhs})", method.suggestion()))
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
