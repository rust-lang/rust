use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for incompatible bit masks in comparisons.
    ///
    /// The formula for detecting if an expression of the type `_ <bit_op> m
    /// <cmp_op> c` (where `<bit_op>` is one of {`&`, `|`} and `<cmp_op>` is one of
    /// {`!=`, `>=`, `>`, `!=`, `>=`, `>`}) can be determined from the following
    /// table:
    ///
    /// |Comparison  |Bit Op|Example      |is always|Formula               |
    /// |------------|------|-------------|---------|----------------------|
    /// |`==` or `!=`| `&`  |`x & 2 == 3` |`false`  |`c & m != c`          |
    /// |`<`  or `>=`| `&`  |`x & 2 < 3`  |`true`   |`m < c`               |
    /// |`>`  or `<=`| `&`  |`x & 1 > 1`  |`false`  |`m <= c`              |
    /// |`==` or `!=`| `\|` |`x \| 1 == 0`|`false`  |`c \| m != c`         |
    /// |`<`  or `>=`| `\|` |`x \| 1 < 1` |`false`  |`m >= c`              |
    /// |`<=` or `>` | `\|` |`x \| 1 > 0` |`true`   |`m > c`               |
    ///
    /// ### Why is this bad?
    /// If the bits that the comparison cares about are always
    /// set to zero or one by the bit mask, the comparison is constant `true` or
    /// `false` (depending on mask, compared value, and operators).
    ///
    /// So the code is actively misleading, and the only reason someone would write
    /// this intentionally is to win an underhanded Rust contest or create a
    /// test-case for this lint.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if (x & 1 == 2) { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BAD_BIT_MASK,
    correctness,
    "expressions of the form `_ & mask == select` that will only ever return `true` or `false`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bit masks in comparisons which can be removed
    /// without changing the outcome. The basic structure can be seen in the
    /// following table:
    ///
    /// |Comparison| Bit Op   |Example     |equals |
    /// |----------|----------|------------|-------|
    /// |`>` / `<=`|`\|` / `^`|`x \| 2 > 3`|`x > 3`|
    /// |`<` / `>=`|`\|` / `^`|`x ^ 1 < 4` |`x < 4`|
    ///
    /// ### Why is this bad?
    /// Not equally evil as [`bad_bit_mask`](#bad_bit_mask),
    /// but still a bit misleading, because the bit mask is ineffective.
    ///
    /// ### Known problems
    /// False negatives: This lint will only match instances
    /// where we have figured out the math (which is for a power-of-two compared
    /// value). This means things like `x | 1 >= 7` (which would be better written
    /// as `x >= 6`) will not be reported (but bit masks like this are fairly
    /// uncommon).
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if (x | 1 > 3) {  }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INEFFECTIVE_BIT_MASK,
    correctness,
    "expressions where a bit mask will be rendered useless by a comparison, e.g., `(x | 1) > 2`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for bit masks that can be replaced by a call
    /// to `trailing_zeros`
    ///
    /// ### Why is this bad?
    /// `x.trailing_zeros() > 4` is much clearer than `x & 15
    /// == 0`
    ///
    /// ### Known problems
    /// llvm generates better code for `x & 15 == 0` on x86
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// if x & 0b1111 == 0 { }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub VERBOSE_BIT_MASK,
    pedantic,
    "expressions where a bit mask is less readable than the corresponding method call"
}

#[derive(Copy, Clone)]
pub struct BitMask {
    verbose_bit_mask_threshold: u64,
}

impl BitMask {
    #[must_use]
    pub fn new(verbose_bit_mask_threshold: u64) -> Self {
        Self {
            verbose_bit_mask_threshold,
        }
    }
}

impl_lint_pass!(BitMask => [BAD_BIT_MASK, INEFFECTIVE_BIT_MASK, VERBOSE_BIT_MASK]);

impl<'tcx> LateLintPass<'tcx> for BitMask {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Binary(cmp, left, right) = &e.kind {
            if cmp.node.is_comparison() {
                if let Some(cmp_opt) = fetch_int_literal(cx, right) {
                    check_compare(cx, left, cmp.node, cmp_opt, e.span);
                } else if let Some(cmp_val) = fetch_int_literal(cx, left) {
                    check_compare(cx, right, invert_cmp(cmp.node), cmp_val, e.span);
                }
            }
        }
        if_chain! {
            if let ExprKind::Binary(op, left, right) = &e.kind;
            if BinOpKind::Eq == op.node;
            if let ExprKind::Binary(op1, left1, right1) = &left.kind;
            if BinOpKind::BitAnd == op1.node;
            if let ExprKind::Lit(lit) = &right1.kind;
            if let LitKind::Int(n, _) = lit.node;
            if let ExprKind::Lit(lit1) = &right.kind;
            if let LitKind::Int(0, _) = lit1.node;
            if n.leading_zeros() == n.count_zeros();
            if n > u128::from(self.verbose_bit_mask_threshold);
            then {
                span_lint_and_then(cx,
                                   VERBOSE_BIT_MASK,
                                   e.span,
                                   "bit mask could be simplified with a call to `trailing_zeros`",
                                   |diag| {
                    let sugg = Sugg::hir(cx, left1, "...").maybe_par();
                    diag.span_suggestion(
                        e.span,
                        "try",
                        format!("{}.trailing_zeros() >= {}", sugg, n.count_ones()),
                        Applicability::MaybeIncorrect,
                    );
                });
            }
        }
    }
}

#[must_use]
fn invert_cmp(cmp: BinOpKind) -> BinOpKind {
    match cmp {
        BinOpKind::Eq => BinOpKind::Eq,
        BinOpKind::Ne => BinOpKind::Ne,
        BinOpKind::Lt => BinOpKind::Gt,
        BinOpKind::Gt => BinOpKind::Lt,
        BinOpKind::Le => BinOpKind::Ge,
        BinOpKind::Ge => BinOpKind::Le,
        _ => BinOpKind::Or, // Dummy
    }
}

fn check_compare(cx: &LateContext<'_>, bit_op: &Expr<'_>, cmp_op: BinOpKind, cmp_value: u128, span: Span) {
    if let ExprKind::Binary(op, left, right) = &bit_op.kind {
        if op.node != BinOpKind::BitAnd && op.node != BinOpKind::BitOr {
            return;
        }
        fetch_int_literal(cx, right)
            .or_else(|| fetch_int_literal(cx, left))
            .map_or((), |mask| check_bit_mask(cx, op.node, cmp_op, mask, cmp_value, span));
    }
}

#[allow(clippy::too_many_lines)]
fn check_bit_mask(
    cx: &LateContext<'_>,
    bit_op: BinOpKind,
    cmp_op: BinOpKind,
    mask_value: u128,
    cmp_value: u128,
    span: Span,
) {
    match cmp_op {
        BinOpKind::Eq | BinOpKind::Ne => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value & cmp_value != cmp_value {
                    if cmp_value != 0 {
                        span_lint(
                            cx,
                            BAD_BIT_MASK,
                            span,
                            &format!(
                                "incompatible bit mask: `_ & {}` can never be equal to `{}`",
                                mask_value, cmp_value
                            ),
                        );
                    }
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value | cmp_value != cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        &format!(
                            "incompatible bit mask: `_ | {}` can never be equal to `{}`",
                            mask_value, cmp_value
                        ),
                    );
                }
            },
            _ => (),
        },
        BinOpKind::Lt | BinOpKind::Ge => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value < cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        &format!(
                            "incompatible bit mask: `_ & {}` will always be lower than `{}`",
                            mask_value, cmp_value
                        ),
                    );
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value >= cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        &format!(
                            "incompatible bit mask: `_ | {}` will never be lower than `{}`",
                            mask_value, cmp_value
                        ),
                    );
                } else {
                    check_ineffective_lt(cx, span, mask_value, cmp_value, "|");
                }
            },
            BinOpKind::BitXor => check_ineffective_lt(cx, span, mask_value, cmp_value, "^"),
            _ => (),
        },
        BinOpKind::Le | BinOpKind::Gt => match bit_op {
            BinOpKind::BitAnd => {
                if mask_value <= cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        &format!(
                            "incompatible bit mask: `_ & {}` will never be higher than `{}`",
                            mask_value, cmp_value
                        ),
                    );
                } else if mask_value == 0 {
                    span_lint(cx, BAD_BIT_MASK, span, "&-masking with zero");
                }
            },
            BinOpKind::BitOr => {
                if mask_value > cmp_value {
                    span_lint(
                        cx,
                        BAD_BIT_MASK,
                        span,
                        &format!(
                            "incompatible bit mask: `_ | {}` will always be higher than `{}`",
                            mask_value, cmp_value
                        ),
                    );
                } else {
                    check_ineffective_gt(cx, span, mask_value, cmp_value, "|");
                }
            },
            BinOpKind::BitXor => check_ineffective_gt(cx, span, mask_value, cmp_value, "^"),
            _ => (),
        },
        _ => (),
    }
}

fn check_ineffective_lt(cx: &LateContext<'_>, span: Span, m: u128, c: u128, op: &str) {
    if c.is_power_of_two() && m < c {
        span_lint(
            cx,
            INEFFECTIVE_BIT_MASK,
            span,
            &format!(
                "ineffective bit mask: `x {} {}` compared to `{}`, is the same as x compared directly",
                op, m, c
            ),
        );
    }
}

fn check_ineffective_gt(cx: &LateContext<'_>, span: Span, m: u128, c: u128, op: &str) {
    if (c + 1).is_power_of_two() && m <= c {
        span_lint(
            cx,
            INEFFECTIVE_BIT_MASK,
            span,
            &format!(
                "ineffective bit mask: `x {} {}` compared to `{}`, is the same as x compared directly",
                op, m, c
            ),
        );
    }
}

fn fetch_int_literal(cx: &LateContext<'_>, lit: &Expr<'_>) -> Option<u128> {
    match constant(cx, cx.typeck_results(), lit)?.0 {
        Constant::Int(n) => Some(n),
        _ => None,
    }
}
