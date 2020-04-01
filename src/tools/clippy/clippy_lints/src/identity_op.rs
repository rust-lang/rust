use rustc_hir::{BinOpKind, Expr, ExprKind};
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for IdentityOp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }
        if let ExprKind::Binary(ref cmp, ref left, ref right) = e.kind {
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

#[allow(clippy::cast_possible_wrap)]
fn check(cx: &LateContext<'_, '_>, e: &Expr<'_>, m: i8, span: Span, arg: Span) {
    if let Some(Constant::Int(v)) = constant_simple(cx, cx.tables, e) {
        let check = match cx.tables.expr_ty(e).kind {
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
