use crate::utils;
use rustc::{declare_tool_lint, hir, lint, lint_array};
use rustc_errors::Applicability;
use std::fmt;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of the `offset` pointer method with a `usize` casted to an
    /// `isize`.
    ///
    /// **Why is this bad?** If we’re always increasing the pointer address, we can avoid the numeric
    /// cast by using the `add` method instead.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
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
    pub PTR_OFFSET_WITH_CAST,
    complexity,
    "unneeded pointer offset cast"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl lint::LintPass for Pass {
    fn get_lints(&self) -> lint::LintArray {
        lint_array!(PTR_OFFSET_WITH_CAST)
    }

    fn name(&self) -> &'static str {
        "PtrOffsetWithCast"
    }
}

impl<'a, 'tcx> lint::LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &lint::LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        // Check if the expressions is a ptr.offset or ptr.wrapping_offset method call
        let (receiver_expr, arg_expr, method) = match expr_as_ptr_offset_call(cx, expr) {
            Some(call_arg) => call_arg,
            None => return,
        };

        // Check if the argument to the method call is a cast from usize
        let cast_lhs_expr = match expr_as_cast_from_usize(cx, arg_expr) {
            Some(cast_lhs_expr) => cast_lhs_expr,
            None => return,
        };

        let msg = format!("use of `{}` with a `usize` casted to an `isize`", method);
        if let Some(sugg) = build_suggestion(cx, method, receiver_expr, cast_lhs_expr) {
            utils::span_lint_and_sugg(
                cx,
                PTR_OFFSET_WITH_CAST,
                expr.span,
                &msg,
                "try",
                sugg,
                Applicability::MachineApplicable,
            );
        } else {
            utils::span_lint(cx, PTR_OFFSET_WITH_CAST, expr.span, &msg);
        }
    }
}

// If the given expression is a cast from a usize, return the lhs of the cast
fn expr_as_cast_from_usize<'a, 'tcx>(
    cx: &lint::LateContext<'a, 'tcx>,
    expr: &'tcx hir::Expr,
) -> Option<&'tcx hir::Expr> {
    if let hir::ExprKind::Cast(ref cast_lhs_expr, _) = expr.node {
        if is_expr_ty_usize(cx, &cast_lhs_expr) {
            return Some(cast_lhs_expr);
        }
    }
    None
}

// If the given expression is a ptr::offset  or ptr::wrapping_offset method call, return the
// receiver, the arg of the method call, and the method.
fn expr_as_ptr_offset_call<'a, 'tcx>(
    cx: &lint::LateContext<'a, 'tcx>,
    expr: &'tcx hir::Expr,
) -> Option<(&'tcx hir::Expr, &'tcx hir::Expr, Method)> {
    if let hir::ExprKind::MethodCall(ref path_segment, _, ref args) = expr.node {
        if is_expr_ty_raw_ptr(cx, &args[0]) {
            if path_segment.ident.name == "offset" {
                return Some((&args[0], &args[1], Method::Offset));
            }
            if path_segment.ident.name == "wrapping_offset" {
                return Some((&args[0], &args[1], Method::WrappingOffset));
            }
        }
    }
    None
}

// Is the type of the expression a usize?
fn is_expr_ty_usize<'a, 'tcx>(cx: &lint::LateContext<'a, 'tcx>, expr: &hir::Expr) -> bool {
    cx.tables.expr_ty(expr) == cx.tcx.types.usize
}

// Is the type of the expression a raw pointer?
fn is_expr_ty_raw_ptr<'a, 'tcx>(cx: &lint::LateContext<'a, 'tcx>, expr: &hir::Expr) -> bool {
    cx.tables.expr_ty(expr).is_unsafe_ptr()
}

fn build_suggestion<'a, 'tcx>(
    cx: &lint::LateContext<'a, 'tcx>,
    method: Method,
    receiver_expr: &hir::Expr,
    cast_lhs_expr: &hir::Expr,
) -> Option<String> {
    let receiver = utils::snippet_opt(cx, receiver_expr.span)?;
    let cast_lhs = utils::snippet_opt(cx, cast_lhs_expr.span)?;
    Some(format!("{}.{}({})", receiver, method.suggestion(), cast_lhs))
}

#[derive(Copy, Clone)]
enum Method {
    Offset,
    WrappingOffset,
}

impl Method {
    fn suggestion(self) -> &'static str {
        match self {
            Method::Offset => "add",
            Method::WrappingOffset => "wrapping_add",
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Method::Offset => write!(f, "offset"),
            Method::WrappingOffset => write!(f, "wrapping_offset"),
        }
    }
}
