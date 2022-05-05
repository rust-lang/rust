use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet;
use rustc_hir::{BinOp, BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

use clippy_utils::consts::{constant_full_int, constant_simple, Constant, FullInt};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{clip, unsext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for identity operations, e.g., `x + 0`.
    ///
    /// ### Why is this bad?
    /// This code can be removed without changing the
    /// meaning. So it just obscures what's going on. Delete it mercilessly.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// x / 1 + 0 * 1 - 0 | 0;
    /// ```
    ///
    /// ### Known problems
    /// False negatives: `f(0 + if b { 1 } else { 2 } + 3);` is reducible to
    /// `f(if b { 1 } else { 2 } + 3);`. But the lint doesn't trigger for the code.
    /// See [#8724](https://github.com/rust-lang/rust-clippy/issues/8724)
    #[clippy::version = "pre 1.29.0"]
    pub IDENTITY_OP,
    complexity,
    "using identity operations, e.g., `x + 0` or `y / 1`"
}

declare_lint_pass!(IdentityOp => [IDENTITY_OP]);

impl<'tcx> LateLintPass<'tcx> for IdentityOp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        if let ExprKind::Binary(cmp, left, right) = &expr.kind {
            if !is_allowed(cx, *cmp, left, right) {
                match cmp.node {
                    BinOpKind::Add | BinOpKind::BitOr | BinOpKind::BitXor => {
                        if reducible_to_right(cx, expr, right) {
                            check(cx, left, 0, expr.span, right.span);
                        }
                        check(cx, right, 0, expr.span, left.span);
                    },
                    BinOpKind::Shl | BinOpKind::Shr | BinOpKind::Sub => {
                        check(cx, right, 0, expr.span, left.span);
                    },
                    BinOpKind::Mul => {
                        if reducible_to_right(cx, expr, right) {
                            check(cx, left, 1, expr.span, right.span);
                        }
                        check(cx, right, 1, expr.span, left.span);
                    },
                    BinOpKind::Div => check(cx, right, 1, expr.span, left.span),
                    BinOpKind::BitAnd => {
                        if reducible_to_right(cx, expr, right) {
                            check(cx, left, -1, expr.span, right.span);
                        }
                        check(cx, right, -1, expr.span, left.span);
                    },
                    BinOpKind::Rem => {
                        // Don't call reducible_to_right because N % N is always reducible to 1
                        check_remainder(cx, left, right, expr.span, left.span);
                    },
                    _ => (),
                }
            }
        }
    }
}

/// Checks if `left op ..right` can be actually reduced to `right`
/// e.g. `0 + if b { 1 } else { 2 } + if b { 3 } else { 4 }`
/// cannot be reduced to `if b { 1 } else { 2 } +  if b { 3 } else { 4 }`
/// See #8724
fn reducible_to_right(cx: &LateContext<'_>, binary: &Expr<'_>, right: &Expr<'_>) -> bool {
    if let ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Block(..) | ExprKind::Loop(..) = right.kind {
        is_toplevel_binary(cx, binary)
    } else {
        true
    }
}

fn is_toplevel_binary(cx: &LateContext<'_>, must_be_binary: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, must_be_binary) && let ExprKind::Binary(..) = &parent.kind {
        false
    } else {
        true
    }
}

fn is_allowed(cx: &LateContext<'_>, cmp: BinOp, left: &Expr<'_>, right: &Expr<'_>) -> bool {
    // This lint applies to integers
    !cx.typeck_results().expr_ty(left).peel_refs().is_integral()
        || !cx.typeck_results().expr_ty(right).peel_refs().is_integral()
        // `1 << 0` is a common pattern in bit manipulation code
        || (cmp.node == BinOpKind::Shl
            && constant_simple(cx, cx.typeck_results(), right) == Some(Constant::Int(0))
            && constant_simple(cx, cx.typeck_results(), left) == Some(Constant::Int(1)))
}

fn check_remainder(cx: &LateContext<'_>, left: &Expr<'_>, right: &Expr<'_>, span: Span, arg: Span) {
    let lhs_const = constant_full_int(cx, cx.typeck_results(), left);
    let rhs_const = constant_full_int(cx, cx.typeck_results(), right);
    if match (lhs_const, rhs_const) {
        (Some(FullInt::S(lv)), Some(FullInt::S(rv))) => lv.abs() < rv.abs(),
        (Some(FullInt::U(lv)), Some(FullInt::U(rv))) => lv < rv,
        _ => return,
    } {
        span_ineffective_operation(cx, span, arg);
    }
}

fn check(cx: &LateContext<'_>, e: &Expr<'_>, m: i8, span: Span, arg: Span) {
    if let Some(Constant::Int(v)) = constant_simple(cx, cx.typeck_results(), e).map(Constant::peel_refs) {
        let check = match *cx.typeck_results().expr_ty(e).peel_refs().kind() {
            ty::Int(ity) => unsext(cx.tcx, -1_i128, ity),
            ty::Uint(uty) => clip(cx.tcx, !0, uty),
            _ => return,
        };
        if match m {
            0 => v == 0,
            -1 => v == check,
            1 => v == 1,
            _ => unreachable!(),
        } {
            span_ineffective_operation(cx, span, arg);
        }
    }
}

fn span_ineffective_operation(cx: &LateContext<'_>, span: Span, arg: Span) {
    span_lint(
        cx,
        IDENTITY_OP,
        span,
        &format!(
            "the operation is ineffective. Consider reducing it to `{}`",
            snippet(cx, arg, "..")
        ),
    );
}
