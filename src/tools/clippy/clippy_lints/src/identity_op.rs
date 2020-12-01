use if_chain::if_chain;
use rustc_hir::{BinOp, BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

use crate::consts::{constant_simple, Constant};
use crate::utils::{clip, snippet, span_lint, unsext};

declare_clippy_lint! {
    /// **What it does:** Checks for identity operations, e.g., `x + 0`.
    ///
    /// **Why is this bad?** This code can be removed without changing the
    /// meaning. So it just obscures what's going on. Delete it mercilessly.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let x = 1;
    /// x / 1 + 0 * 1 - 0 | 0;
    /// ```
    pub IDENTITY_OP,
    complexity,
    "using identity operations, e.g., `x + 0` or `y / 1`"
}

declare_lint_pass!(IdentityOp => [IDENTITY_OP]);

impl<'tcx> LateLintPass<'tcx> for IdentityOp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }
        if let ExprKind::Binary(cmp, ref left, ref right) = e.kind {
            if is_allowed(cx, cmp, left, right) {
                return;
            }
            match cmp.node {
                BinOpKind::Add | BinOpKind::BitOr | BinOpKind::BitXor => {
                    check(cx, left, 0, e.span, right.span);
                    check(cx, right, 0, e.span, left.span);
                },
                BinOpKind::Shl | BinOpKind::Shr | BinOpKind::Sub => check(cx, right, 0, e.span, left.span),
                BinOpKind::Mul => {
                    check(cx, left, 1, e.span, right.span);
                    check(cx, right, 1, e.span, left.span);
                },
                BinOpKind::Div => check(cx, right, 1, e.span, left.span),
                BinOpKind::BitAnd => {
                    check(cx, left, -1, e.span, right.span);
                    check(cx, right, -1, e.span, left.span);
                },
                _ => (),
            }
        }
    }
}

fn is_allowed(cx: &LateContext<'_>, cmp: BinOp, left: &Expr<'_>, right: &Expr<'_>) -> bool {
    // `1 << 0` is a common pattern in bit manipulation code
    if_chain! {
        if let BinOpKind::Shl = cmp.node;
        if let Some(Constant::Int(0)) = constant_simple(cx, cx.typeck_results(), right);
        if let Some(Constant::Int(1)) = constant_simple(cx, cx.typeck_results(), left);
        then {
            return true;
        }
    }

    false
}

#[allow(clippy::cast_possible_wrap)]
fn check(cx: &LateContext<'_>, e: &Expr<'_>, m: i8, span: Span, arg: Span) {
    if let Some(Constant::Int(v)) = constant_simple(cx, cx.typeck_results(), e) {
        let check = match *cx.typeck_results().expr_ty(e).kind() {
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
    }
}
