use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::LitKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_hir::ExprKind;
use rustc_hir::PatKind;
use rustc_hir::RangeEnd;
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Looks for combined OR patterns that are all contained in a specific range,
    /// e.g. `6 | 4 | 5 | 9 | 7 | 8` can be rewritten as `4..=9`.
    ///
    /// ### Why is this bad?
    /// Using an explicit range is more concise and easier to read.
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

fn expr_as_u128(expr: &Expr<'_>) -> Option<u128> {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Int(num, _) = lit.node
    {
        Some(num)
    } else {
        None
    }
}

impl LateLintPass<'_> for ManualRangePatterns {
    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &'_ rustc_hir::Pat<'_>) {
        if in_external_macro(cx.sess(), pat.span) {
            return;
        }

        // a pattern like 1 | 2 seems fine, lint if there are at least 3 alternatives
        if let PatKind::Or(pats) = pat.kind
            && pats.len() >= 3
        {
            let mut min = u128::MAX;
            let mut max = 0;
            let mut numbers_found = FxHashSet::default();
            let mut ranges_found = Vec::new();

            for pat in pats {
                if let PatKind::Lit(lit) = pat.kind
                    && let Some(num) = expr_as_u128(lit)
                {
                    numbers_found.insert(num);

                    min = min.min(num);
                    max = max.max(num);
                } else if let PatKind::Range(Some(left), Some(right), end) = pat.kind
                    && let Some(left) = expr_as_u128(left)
                    && let Some(right) = expr_as_u128(right)
                    && right >= left
                {
                    min = min.min(left);
                    max = max.max(right);
                    ranges_found.push(left..=match end {
                        RangeEnd::Included => right,
                        RangeEnd::Excluded => right - 1,
                    });
                } else {
                    return;
                }
            }

            let contains_whole_range = 'contains: {
                let mut num = min;
                while num <= max {
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
                        break 'contains false;
                    }
                }
                break 'contains true;
            };

            if contains_whole_range {
                span_lint_and_sugg(
                    cx,
                    MANUAL_RANGE_PATTERNS,
                    pat.span,
                    "this OR pattern can be rewritten using a range",
                    "try",
                    format!("{min}..={max}"),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
