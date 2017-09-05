use rustc::hir::*;
use rustc::lint::*;
use utils::{implements_trait, is_copy, multispan_sugg, snippet, span_lint, span_lint_and_then, SpanlessEq};

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

/// **What it does:** Checks for arguments to `==` which have their address
/// taken to satisfy a bound
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

#[derive(Copy, Clone)]
pub struct EqOp;

impl LintPass for EqOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(EQ_OP, OP_REF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EqOp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprBinary(ref op, ref left, ref right) = e.node {
            if is_valid_operator(op) && SpanlessEq::new(cx).ignore_fn().eq_expr(left, right) {
                span_lint(
                    cx,
                    EQ_OP,
                    e.span,
                    &format!("equal expressions as operands to `{}`", op.node.as_str()),
                );
                return;
            }
            let (trait_id, requires_ref) = match op.node {
                BiAdd => (cx.tcx.lang_items.add_trait(), false),
                BiSub => (cx.tcx.lang_items.sub_trait(), false),
                BiMul => (cx.tcx.lang_items.mul_trait(), false),
                BiDiv => (cx.tcx.lang_items.div_trait(), false),
                BiRem => (cx.tcx.lang_items.rem_trait(), false),
                // don't lint short circuiting ops
                BiAnd | BiOr => return,
                BiBitXor => (cx.tcx.lang_items.bitxor_trait(), false),
                BiBitAnd => (cx.tcx.lang_items.bitand_trait(), false),
                BiBitOr => (cx.tcx.lang_items.bitor_trait(), false),
                BiShl => (cx.tcx.lang_items.shl_trait(), false),
                BiShr => (cx.tcx.lang_items.shr_trait(), false),
                BiNe | BiEq => (cx.tcx.lang_items.eq_trait(), true),
                BiLt | BiLe | BiGe | BiGt => (cx.tcx.lang_items.ord_trait(), true),
            };
            if let Some(trait_id) = trait_id {
                #[allow(match_same_arms)]
                match (&left.node, &right.node) {
                    // do not suggest to dereference literals
                    (&ExprLit(..), _) | (_, &ExprLit(..)) => {},
                    // &foo == &bar
                    (&ExprAddrOf(_, ref l), &ExprAddrOf(_, ref r)) => {
                        let lty = cx.tables.expr_ty(l);
                        let rty = cx.tables.expr_ty(r);
                        let lcpy = is_copy(cx, lty);
                        let rcpy = is_copy(cx, rty);
                        // either operator autorefs or both args are copyable
                        if (requires_ref || (lcpy && rcpy)) && implements_trait(cx, lty, trait_id, &[rty]) {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of both operands",
                                |db| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    multispan_sugg(
                                        db,
                                        "use the values directly".to_string(),
                                        vec![(left.span, lsnip), (right.span, rsnip)],
                                    );
                                },
                            )
                        } else if lcpy && !rcpy && implements_trait(cx, lty, trait_id, &[cx.tables.expr_ty(right)]) {
                            span_lint_and_then(cx, OP_REF, e.span, "needlessly taken reference of left operand", |db| {
                                let lsnip = snippet(cx, l.span, "...").to_string();
                                db.span_suggestion(left.span, "use the left value directly", lsnip);
                            })
                        } else if !lcpy && rcpy && implements_trait(cx, cx.tables.expr_ty(left), trait_id, &[rty]) {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of right operand",
                                |db| {
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    db.span_suggestion(right.span, "use the right value directly", rsnip);
                                },
                            )
                        }
                    },
                    // &foo == bar
                    (&ExprAddrOf(_, ref l), _) => {
                        let lty = cx.tables.expr_ty(l);
                        let lcpy = is_copy(cx, lty);
                        if (requires_ref || lcpy) && implements_trait(cx, lty, trait_id, &[cx.tables.expr_ty(right)]) {
                            span_lint_and_then(cx, OP_REF, e.span, "needlessly taken reference of left operand", |db| {
                                let lsnip = snippet(cx, l.span, "...").to_string();
                                db.span_suggestion(left.span, "use the left value directly", lsnip);
                            })
                        }
                    },
                    // foo == &bar
                    (_, &ExprAddrOf(_, ref r)) => {
                        let rty = cx.tables.expr_ty(r);
                        let rcpy = is_copy(cx, rty);
                        if (requires_ref || rcpy) && implements_trait(cx, cx.tables.expr_ty(left), trait_id, &[rty]) {
                            span_lint_and_then(cx, OP_REF, e.span, "taken reference of right operand", |db| {
                                let rsnip = snippet(cx, r.span, "...").to_string();
                                db.span_suggestion(right.span, "use the right value directly", rsnip);
                            })
                        }
                    },
                    _ => {},
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
