use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PatKind, RangeEnd, UnOp};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{Span, DUMMY_SP};

declare_clippy_lint! {
    /// ### What it does
    /// Looks for combined OR patterns that are all contained in a specific range,
    /// e.g. `6 | 4 | 5 | 9 | 7 | 8` can be rewritten as `4..=9`.
    ///
    /// ### Why is this bad?
    /// Using an explicit range is more concise and easier to read.
    ///
    /// ### Known issues
    /// This lint intentionally does not handle numbers greater than `i128::MAX` for `u128` literals
    /// in order to support negative numbers.
    ///
    /// ### Example
    /// ```rust
    /// let x = 6;
    /// let foo = matches!(x, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10);
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = 6;
    /// let foo = matches!(x, 1..=10);
    /// ```
    #[clippy::version = "1.72.0"]
    pub MANUAL_RANGE_PATTERNS,
    complexity,
    "manually writing range patterns using a combined OR pattern (`|`)"
}
declare_lint_pass!(ManualRangePatterns => [MANUAL_RANGE_PATTERNS]);

fn expr_as_i128(expr: &Expr<'_>) -> Option<i128> {
    if let ExprKind::Unary(UnOp::Neg, expr) = expr.kind {
        expr_as_i128(expr).map(|num| -num)
    } else if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Int(num, _) = lit.node
    {
        // Intentionally not handling numbers greater than i128::MAX (for u128 literals) for now.
        num.try_into().ok()
    } else {
        None
    }
}

#[derive(Copy, Clone)]
struct Num {
    val: i128,
    span: Span,
}

impl Num {
    fn new(expr: &Expr<'_>) -> Option<Self> {
        Some(Self {
            val: expr_as_i128(expr)?,
            span: expr.span,
        })
    }

    fn dummy(val: i128) -> Self {
        Self { val, span: DUMMY_SP }
    }

    fn min(self, other: Self) -> Self {
        if self.val < other.val { self } else { other }
    }
}

impl LateLintPass<'_> for ManualRangePatterns {
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &'_ rustc_hir::Pat<'_>) {
        if in_external_macro(cx.sess(), pat.span) {
            return;
        }

        // a pattern like 1 | 2 seems fine, lint if there are at least 3 alternatives
        // or at least one range
        if let PatKind::Or(pats) = pat.kind
            && (pats.len() >= 3 || pats.iter().any(|p| matches!(p.kind, PatKind::Range(..))))
        {
            let mut min = Num::dummy(i128::MAX);
            let mut max = Num::dummy(i128::MIN);
            let mut range_kind = RangeEnd::Included;
            let mut numbers_found = FxHashSet::default();
            let mut ranges_found = Vec::new();

            for pat in pats {
                if let PatKind::Lit(lit) = pat.kind
                    && let Some(num) = Num::new(lit)
                {
                    numbers_found.insert(num.val);

                    min = min.min(num);
                    if num.val >= max.val {
                        max = num;
                        range_kind = RangeEnd::Included;
                    }
                } else if let PatKind::Range(Some(left), Some(right), end) = pat.kind
                    && let Some(left) = Num::new(left)
                    && let Some(mut right) = Num::new(right)
                {
                    if let RangeEnd::Excluded = end {
                        right.val -= 1;
                    }

                    min = min.min(left);
                    if right.val > max.val {
                        max = right;
                        range_kind = end;
                    }
                    ranges_found.push(left.val..=right.val);
                } else {
                    return;
                }
            }

            let mut num = min.val;
            while num <= max.val {
                if numbers_found.contains(&num) {
                    num += 1;
                }
                // Given a list of (potentially overlapping) ranges like:
                // 1..=5, 3..=7, 6..=10
                // We want to find the range with the highest end that still contains the current number
                else if let Some(range) = ranges_found
                    .iter()
                    .filter(|range| range.contains(&num))
                    .max_by_key(|range| range.end())
                {
                    num = range.end() + 1;
                } else {
                    return;
                }
            }

            span_lint_and_then(
                cx,
                MANUAL_RANGE_PATTERNS,
                pat.span,
                "this OR pattern can be rewritten using a range",
                |diag| {
                    if let Some(min) = snippet_opt(cx, min.span)
                        && let Some(max) = snippet_opt(cx, max.span)
                    {
                        diag.span_suggestion(
                            pat.span,
                            "try",
                            format!("{min}{range_kind}{max}"),
                            Applicability::MachineApplicable,
                        );
                    }
                },
            );
        }
    }
}
