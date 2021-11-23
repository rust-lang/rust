use clippy_utils::diagnostics::{multispan_sugg, span_lint, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::ty::{implements_trait, is_copy};
use clippy_utils::{ast_utils::is_useless_with_eq_exprs, eq_expr_value, higher, is_expn_of, is_in_test_function};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for equal operands to comparison, logical and
    /// bitwise, difference and division binary operators (`==`, `>`, etc., `&&`,
    /// `||`, `&`, `|`, `^`, `-` and `/`).
    ///
    /// ### Why is this bad?
    /// This is usually just a typo or a copy and paste error.
    ///
    /// ### Known problems
    /// False negatives: We had some false positives regarding
    /// calls (notably [racer](https://github.com/phildawes/racer) had one instance
    /// of `x.pop() && x.pop()`), so we removed matching any function or method
    /// calls. We may introduce a list of known pure functions in the future.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if x + 1 == x + 1 {}
    /// ```
    /// or
    /// ```rust
    /// # let a = 3;
    /// # let b = 4;
    /// assert_eq!(a, a);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EQ_OP,
    correctness,
    "equal operands on both sides of a comparison or bitwise combination (e.g., `x == x`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for arguments to `==` which have their address
    /// taken to satisfy a bound
    /// and suggests to dereference the other argument instead
    ///
    /// ### Why is this bad?
    /// It is more idiomatic to dereference the other argument.
    ///
    /// ### Known problems
    /// None
    ///
    /// ### Example
    /// ```ignore
    /// // Bad
    /// &x == y
    ///
    /// // Good
    /// x == *y
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OP_REF,
    style,
    "taking a reference to satisfy the type constraints on `==`"
}

declare_lint_pass!(EqOp => [EQ_OP, OP_REF]);

const ASSERT_MACRO_NAMES: [&str; 4] = ["assert_eq", "assert_ne", "debug_assert_eq", "debug_assert_ne"];

impl<'tcx> LateLintPass<'tcx> for EqOp {
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Block(block, _) = e.kind {
            for stmt in block.stmts {
                for amn in &ASSERT_MACRO_NAMES {
                    if_chain! {
                        if is_expn_of(stmt.span, amn).is_some();
                        if let StmtKind::Semi(matchexpr) = stmt.kind;
                        if let Some(macro_args) = higher::extract_assert_macro_args(matchexpr);
                        if macro_args.len() == 2;
                        let (lhs, rhs) = (macro_args[0], macro_args[1]);
                        if eq_expr_value(cx, lhs, rhs);
                        if !is_in_test_function(cx.tcx, e.hir_id);
                        then {
                            span_lint(
                                cx,
                                EQ_OP,
                                lhs.span.to(rhs.span),
                                &format!("identical args used in this `{}!` macro call", amn),
                            );
                        }
                    }
                }
            }
        }
        if let ExprKind::Binary(op, left, right) = e.kind {
            if e.span.from_expansion() {
                return;
            }
            let macro_with_not_op = |expr_kind: &ExprKind<'_>| {
                if let ExprKind::Unary(_, expr) = *expr_kind {
                    expr.span.from_expansion()
                } else {
                    false
                }
            };
            if macro_with_not_op(&left.kind) || macro_with_not_op(&right.kind) {
                return;
            }
            if is_useless_with_eq_exprs(op.node.into())
                && eq_expr_value(cx, left, right)
                && !is_in_test_function(cx.tcx, e.hir_id)
            {
                span_lint(
                    cx,
                    EQ_OP,
                    e.span,
                    &format!("equal expressions as operands to `{}`", op.node.as_str()),
                );
                return;
            }
            let (trait_id, requires_ref) = match op.node {
                BinOpKind::Add => (cx.tcx.lang_items().add_trait(), false),
                BinOpKind::Sub => (cx.tcx.lang_items().sub_trait(), false),
                BinOpKind::Mul => (cx.tcx.lang_items().mul_trait(), false),
                BinOpKind::Div => (cx.tcx.lang_items().div_trait(), false),
                BinOpKind::Rem => (cx.tcx.lang_items().rem_trait(), false),
                // don't lint short circuiting ops
                BinOpKind::And | BinOpKind::Or => return,
                BinOpKind::BitXor => (cx.tcx.lang_items().bitxor_trait(), false),
                BinOpKind::BitAnd => (cx.tcx.lang_items().bitand_trait(), false),
                BinOpKind::BitOr => (cx.tcx.lang_items().bitor_trait(), false),
                BinOpKind::Shl => (cx.tcx.lang_items().shl_trait(), false),
                BinOpKind::Shr => (cx.tcx.lang_items().shr_trait(), false),
                BinOpKind::Ne | BinOpKind::Eq => (cx.tcx.lang_items().eq_trait(), true),
                BinOpKind::Lt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Gt => {
                    (cx.tcx.lang_items().partial_ord_trait(), true)
                },
            };
            if let Some(trait_id) = trait_id {
                #[allow(clippy::match_same_arms)]
                match (&left.kind, &right.kind) {
                    // do not suggest to dereference literals
                    (&ExprKind::Lit(..), _) | (_, &ExprKind::Lit(..)) => {},
                    // &foo == &bar
                    (&ExprKind::AddrOf(BorrowKind::Ref, _, l), &ExprKind::AddrOf(BorrowKind::Ref, _, r)) => {
                        let lty = cx.typeck_results().expr_ty(l);
                        let rty = cx.typeck_results().expr_ty(r);
                        let lcpy = is_copy(cx, lty);
                        let rcpy = is_copy(cx, rty);
                        // either operator autorefs or both args are copyable
                        if (requires_ref || (lcpy && rcpy)) && implements_trait(cx, lty, trait_id, &[rty.into()]) {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of both operands",
                                |diag| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    multispan_sugg(
                                        diag,
                                        "use the values directly",
                                        vec![(left.span, lsnip), (right.span, rsnip)],
                                    );
                                },
                            );
                        } else if lcpy
                            && !rcpy
                            && implements_trait(cx, lty, trait_id, &[cx.typeck_results().expr_ty(right).into()])
                        {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of left operand",
                                |diag| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    diag.span_suggestion(
                                        left.span,
                                        "use the left value directly",
                                        lsnip,
                                        Applicability::MaybeIncorrect, // FIXME #2597
                                    );
                                },
                            );
                        } else if !lcpy
                            && rcpy
                            && implements_trait(cx, cx.typeck_results().expr_ty(left), trait_id, &[rty.into()])
                        {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of right operand",
                                |diag| {
                                    let rsnip = snippet(cx, r.span, "...").to_string();
                                    diag.span_suggestion(
                                        right.span,
                                        "use the right value directly",
                                        rsnip,
                                        Applicability::MaybeIncorrect, // FIXME #2597
                                    );
                                },
                            );
                        }
                    },
                    // &foo == bar
                    (&ExprKind::AddrOf(BorrowKind::Ref, _, l), _) => {
                        let lty = cx.typeck_results().expr_ty(l);
                        let lcpy = is_copy(cx, lty);
                        if (requires_ref || lcpy)
                            && implements_trait(cx, lty, trait_id, &[cx.typeck_results().expr_ty(right).into()])
                        {
                            span_lint_and_then(
                                cx,
                                OP_REF,
                                e.span,
                                "needlessly taken reference of left operand",
                                |diag| {
                                    let lsnip = snippet(cx, l.span, "...").to_string();
                                    diag.span_suggestion(
                                        left.span,
                                        "use the left value directly",
                                        lsnip,
                                        Applicability::MaybeIncorrect, // FIXME #2597
                                    );
                                },
                            );
                        }
                    },
                    // foo == &bar
                    (_, &ExprKind::AddrOf(BorrowKind::Ref, _, r)) => {
                        let rty = cx.typeck_results().expr_ty(r);
                        let rcpy = is_copy(cx, rty);
                        if (requires_ref || rcpy)
                            && implements_trait(cx, cx.typeck_results().expr_ty(left), trait_id, &[rty.into()])
                        {
                            span_lint_and_then(cx, OP_REF, e.span, "taken reference of right operand", |diag| {
                                let rsnip = snippet(cx, r.span, "...").to_string();
                                diag.span_suggestion(
                                    right.span,
                                    "use the right value directly",
                                    rsnip,
                                    Applicability::MaybeIncorrect, // FIXME #2597
                                );
                            });
                        }
                    },
                    _ => {},
                }
            }
        }
    }
}
