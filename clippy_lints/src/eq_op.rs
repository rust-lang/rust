use rustc::hir::*;
use rustc::lint::*;
use utils::{SpanlessEq, span_lint, span_lint_and_then, multispan_sugg, snippet};
use utils::sugg::Sugg;

/// **What it does:** Checks for equal operands to comparison, logical and
/// bitwise, difference and division binary operators (`==`, `>`, etc., `&&`,
/// `||`, `&`, `|`, `^`, `-` and `/`).
///
/// **Why is this bad?** This is usually just a typo or a copy and paste error.
///
/// **Known problems:** False negatives: We had some false positives regarding
/// calls (notably [racer](https://github.com/phildawes/racer) had one instance
/// of `x.pop() && x.pop()`), so we removed matching any function or method
/// calls. We may introduce a whitelist of known pure functions in the future.
///
/// **Example:**
/// ```rust
/// x + 1 == x + 1
/// ```
declare_lint! {
    pub EQ_OP,
    Warn,
    "equal operands on both sides of a comparison or bitwise combination (e.g. `x == x`)"
}

/// **What it does:** Checks for arguments to `==` which have their address taken to satisfy a bound
/// and suggests to dereference the other argument instead
///
/// **Why is this bad?** It is more idiomatic to dereference the other argument.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// &x == y
/// ```
declare_lint! {
    pub OP_REF,
    Warn,
    "taking a reference to satisfy the type constraints on `==`"
}

#[derive(Copy,Clone)]
pub struct EqOp;

impl LintPass for EqOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(EQ_OP, OP_REF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EqOp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprBinary(ref op, ref left, ref right) = e.node {
            if is_valid_operator(op) {
                if SpanlessEq::new(cx).ignore_fn().eq_expr(left, right) {
                    span_lint(cx,
                            EQ_OP,
                            e.span,
                            &format!("equal expressions as operands to `{}`", op.node.as_str()));
                } else {
                    match (&left.node, &right.node) {
                        (&ExprAddrOf(_, ref l), &ExprAddrOf(_, ref r)) => {
                            span_lint_and_then(cx,
                                OP_REF,
                                e.span,
                                "taken reference of both operands, which is done automatically by the operator anyway",
                                |db| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    multispan_sugg(db,
                                                "use the values directly".to_string(),
                                                vec![(left.span, lsnip),
                                                     (right.span, rsnip)]);
                                }
                            )
                        }
                        (&ExprAddrOf(_, ref l), _) => {
                            span_lint_and_then(cx,
                                OP_REF,
                                e.span,
                                "taken reference of left operand",
                                |db| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    let rsnip = Sugg::hir(cx, right, "...").deref().to_string();
                                    multispan_sugg(db,
                                                "dereference the right operand instead".to_string(),
                                                vec![(left.span, lsnip),
                                                     (right.span, rsnip)]);
                                }
                            )
                        }
                        (_, &ExprAddrOf(_, ref r)) => {
                            span_lint_and_then(cx,
                                OP_REF,
                                e.span,
                                "taken reference of right operand",
                                |db| {
                                    let lsnip = Sugg::hir(cx, left, "...").deref().to_string();
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    multispan_sugg(db,
                                                "dereference the left operand instead".to_string(),
                                                vec![(left.span, lsnip),
                                                     (right.span, rsnip)]);
                                }
                            )
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}


fn is_valid_operator(op: &BinOp) -> bool {
    match op.node {
        BiSub | BiDiv | BiEq | BiLt | BiLe | BiGt | BiGe | BiNe | BiAnd | BiOr | BiBitXor | BiBitAnd | BiBitOr => true,
        _ => false,
    }
}
