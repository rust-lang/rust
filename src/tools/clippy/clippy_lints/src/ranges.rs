use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{SpanRangeExt, snippet, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, higher, is_in_const_context, is_integer_const, path_to_local};
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, HirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use std::cmp::Ordering;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for exclusive ranges where 1 is added to the
    /// upper bound, e.g., `x..(y+1)`.
    ///
    /// ### Why is this bad?
    /// The code is more readable with an inclusive range
    /// like `x..=y`.
    ///
    /// ### Known problems
    /// Will add unnecessary pair of parentheses when the
    /// expression is not wrapped in a pair but starts with an opening parenthesis
    /// and ends with a closing one.
    /// I.e., `let _ = (f()+1)..(f()+1)` results in `let _ = ((f()+1)..=f())`.
    ///
    /// Also in many cases, inclusive ranges are still slower to run than
    /// exclusive ranges, because they essentially add an extra branch that
    /// LLVM may fail to hoist out of the loop.
    ///
    /// This will cause a warning that cannot be fixed if the consumer of the
    /// range only accepts a specific range type, instead of the generic
    /// `RangeBounds` trait
    /// ([#3307](https://github.com/rust-lang/rust-clippy/issues/3307)).
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 0;
    /// # let y = 1;
    /// for i in x..(y+1) {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = 0;
    /// # let y = 1;
    /// for i in x..=y {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_PLUS_ONE,
    pedantic,
    "`x..(y+1)` reads better as `x..=y`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for inclusive ranges where 1 is subtracted from
    /// the upper bound, e.g., `x..=(y-1)`.
    ///
    /// ### Why is this bad?
    /// The code is more readable with an exclusive range
    /// like `x..y`.
    ///
    /// ### Known problems
    /// This will cause a warning that cannot be fixed if
    /// the consumer of the range only accepts a specific range type, instead of
    /// the generic `RangeBounds` trait
    /// ([#3307](https://github.com/rust-lang/rust-clippy/issues/3307)).
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 0;
    /// # let y = 1;
    /// for i in x..=(y-1) {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let x = 0;
    /// # let y = 1;
    /// for i in x..y {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_MINUS_ONE,
    pedantic,
    "`x..=(y-1)` reads better as `x..y`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for range expressions `x..y` where both `x` and `y`
    /// are constant and `x` is greater to `y`. Also triggers if `x` is equal to `y` when they are conditions to a `for` loop.
    ///
    /// ### Why is this bad?
    /// Empty ranges yield no values so iterating them is a no-op.
    /// Moreover, trying to use a reversed range to index a slice will panic at run-time.
    ///
    /// ### Example
    /// ```rust,no_run
    /// fn main() {
    ///     (10..=0).for_each(|x| println!("{}", x));
    ///
    ///     let arr = [1, 2, 3, 4, 5];
    ///     let sub = &arr[3..1];
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn main() {
    ///     (0..=10).rev().for_each(|x| println!("{}", x));
    ///
    ///     let arr = [1, 2, 3, 4, 5];
    ///     let sub = &arr[1..3];
    /// }
    /// ```
    #[clippy::version = "1.45.0"]
    pub REVERSED_EMPTY_RANGES,
    correctness,
    "reversing the limits of range expressions, resulting in empty ranges"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions like `x >= 3 && x < 8` that could
    /// be more readably expressed as `(3..8).contains(x)`.
    ///
    /// ### Why is this bad?
    /// `contains` expresses the intent better and has less
    /// failure modes (such as fencepost errors or using `||` instead of `&&`).
    ///
    /// ### Example
    /// ```no_run
    /// // given
    /// let x = 6;
    ///
    /// assert!(x >= 3 && x < 8);
    /// ```
    /// Use instead:
    /// ```no_run
    ///# let x = 6;
    /// assert!((3..8).contains(&x));
    /// ```
    #[clippy::version = "1.49.0"]
    pub MANUAL_RANGE_CONTAINS,
    style,
    "manually reimplementing {`Range`, `RangeInclusive`}`::contains`"
}

pub struct Ranges {
    msrv: Msrv,
}

impl Ranges {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(Ranges => [
    RANGE_PLUS_ONE,
    RANGE_MINUS_ONE,
    REVERSED_EMPTY_RANGES,
    MANUAL_RANGE_CONTAINS,
]);

impl<'tcx> LateLintPass<'tcx> for Ranges {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, l, r) = expr.kind
            && self.msrv.meets(cx, msrvs::RANGE_CONTAINS)
        {
            check_possible_range_contains(cx, op.node, l, r, expr, expr.span);
        }

        check_exclusive_range_plus_one(cx, expr);
        check_inclusive_range_minus_one(cx, expr);
        check_reversed_empty_range(cx, expr);
    }
}

fn check_possible_range_contains(
    cx: &LateContext<'_>,
    op: BinOpKind,
    left: &Expr<'_>,
    right: &Expr<'_>,
    expr: &Expr<'_>,
    span: Span,
) {
    if is_in_const_context(cx) {
        return;
    }

    let combine_and = match op {
        BinOpKind::And | BinOpKind::BitAnd => true,
        BinOpKind::Or | BinOpKind::BitOr => false,
        _ => return,
    };
    // value, name, order (higher/lower), inclusiveness
    if let (Some(l), Some(r)) = (check_range_bounds(cx, left), check_range_bounds(cx, right)) {
        // we only lint comparisons on the same name and with different
        // direction
        if l.id != r.id || l.ord == r.ord {
            return;
        }
        let ord = Constant::partial_cmp(cx.tcx, cx.typeck_results().expr_ty(l.expr), &l.val, &r.val);
        if combine_and && ord == Some(r.ord) {
            // order lower bound and upper bound
            let (l_span, u_span, l_inc, u_inc) = if r.ord == Ordering::Less {
                (l.val_span, r.val_span, l.inc, r.inc)
            } else {
                (r.val_span, l.val_span, r.inc, l.inc)
            };
            // we only lint inclusive lower bounds
            if !l_inc {
                return;
            }
            let (range_type, range_op) = if u_inc {
                ("RangeInclusive", "..=")
            } else {
                ("Range", "..")
            };
            let mut applicability = Applicability::MachineApplicable;
            let name = snippet_with_applicability(cx, l.name_span, "_", &mut applicability);
            let lo = snippet_with_applicability(cx, l_span, "_", &mut applicability);
            let hi = snippet_with_applicability(cx, u_span, "_", &mut applicability);
            let space = if lo.ends_with('.') { " " } else { "" };
            span_lint_and_sugg(
                cx,
                MANUAL_RANGE_CONTAINS,
                span,
                format!("manual `{range_type}::contains` implementation"),
                "use",
                format!("({lo}{space}{range_op}{hi}).contains(&{name})"),
                applicability,
            );
        } else if !combine_and && ord == Some(l.ord) {
            // `!_.contains(_)`
            // order lower bound and upper bound
            let (l_span, u_span, l_inc, u_inc) = if l.ord == Ordering::Less {
                (l.val_span, r.val_span, l.inc, r.inc)
            } else {
                (r.val_span, l.val_span, r.inc, l.inc)
            };
            if l_inc {
                return;
            }
            let (range_type, range_op) = if u_inc {
                ("Range", "..")
            } else {
                ("RangeInclusive", "..=")
            };
            let mut applicability = Applicability::MachineApplicable;
            let name = snippet_with_applicability(cx, l.name_span, "_", &mut applicability);
            let lo = snippet_with_applicability(cx, l_span, "_", &mut applicability);
            let hi = snippet_with_applicability(cx, u_span, "_", &mut applicability);
            let space = if lo.ends_with('.') { " " } else { "" };
            span_lint_and_sugg(
                cx,
                MANUAL_RANGE_CONTAINS,
                span,
                format!("manual `!{range_type}::contains` implementation"),
                "use",
                format!("!({lo}{space}{range_op}{hi}).contains(&{name})"),
                applicability,
            );
        }
    }

    // If the LHS is the same operator, we have to recurse to get the "real" RHS, since they have
    // the same operator precedence
    if let ExprKind::Binary(ref lhs_op, _left, new_lhs) = left.kind
        && op == lhs_op.node
        && let new_span = Span::new(new_lhs.span.lo(), right.span.hi(), expr.span.ctxt(), expr.span.parent())
        && new_span.check_source_text(cx, |src| {
            // Do not continue if we have mismatched number of parens, otherwise the suggestion is wrong
            src.matches('(').count() == src.matches(')').count()
        })
    {
        check_possible_range_contains(cx, op, new_lhs, right, expr, new_span);
    }
}

struct RangeBounds<'a, 'tcx> {
    val: Constant<'tcx>,
    expr: &'a Expr<'a>,
    id: HirId,
    name_span: Span,
    val_span: Span,
    ord: Ordering,
    inc: bool,
}

// Takes a binary expression such as x <= 2 as input
// Breaks apart into various pieces, such as the value of the number,
// hir id of the variable, and direction/inclusiveness of the operator
fn check_range_bounds<'a, 'tcx>(cx: &'a LateContext<'tcx>, ex: &'a Expr<'_>) -> Option<RangeBounds<'a, 'tcx>> {
    if let ExprKind::Binary(ref op, l, r) = ex.kind {
        let (inclusive, ordering) = match op.node {
            BinOpKind::Gt => (false, Ordering::Greater),
            BinOpKind::Ge => (true, Ordering::Greater),
            BinOpKind::Lt => (false, Ordering::Less),
            BinOpKind::Le => (true, Ordering::Less),
            _ => return None,
        };
        if let Some(id) = path_to_local(l) {
            if let Some(c) = ConstEvalCtxt::new(cx).eval(r) {
                return Some(RangeBounds {
                    val: c,
                    expr: r,
                    id,
                    name_span: l.span,
                    val_span: r.span,
                    ord: ordering,
                    inc: inclusive,
                });
            }
        } else if let Some(id) = path_to_local(r)
            && let Some(c) = ConstEvalCtxt::new(cx).eval(l)
        {
            return Some(RangeBounds {
                val: c,
                expr: l,
                id,
                name_span: r.span,
                val_span: l.span,
                ord: ordering.reverse(),
                inc: inclusive,
            });
        }
    }
    None
}

// exclusive range plus one: `x..(y+1)`
fn check_exclusive_range_plus_one(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if expr.span.can_be_used_for_suggestions()
        && let Some(higher::Range {
            start,
            end: Some(end),
            limits: RangeLimits::HalfOpen,
        }) = higher::Range::hir(expr)
        && let Some(y) = y_plus_one(cx, end)
    {
        let span = expr.span;
        span_lint_and_then(
            cx,
            RANGE_PLUS_ONE,
            span,
            "an inclusive range would be more readable",
            |diag| {
                let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").maybe_paren().to_string());
                let end = Sugg::hir(cx, y, "y").maybe_paren();
                match span.with_source_text(cx, |src| src.starts_with('(') && src.ends_with(')')) {
                    Some(true) => {
                        diag.span_suggestion(span, "use", format!("({start}..={end})"), Applicability::MaybeIncorrect);
                    },
                    Some(false) => {
                        diag.span_suggestion(
                            span,
                            "use",
                            format!("{start}..={end}"),
                            Applicability::MachineApplicable, // snippet
                        );
                    },
                    None => {},
                }
            },
        );
    }
}

// inclusive range minus one: `x..=(y-1)`
fn check_inclusive_range_minus_one(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if expr.span.can_be_used_for_suggestions()
        && let Some(higher::Range {
            start,
            end: Some(end),
            limits: RangeLimits::Closed,
        }) = higher::Range::hir(expr)
        && let Some(y) = y_minus_one(cx, end)
    {
        span_lint_and_then(
            cx,
            RANGE_MINUS_ONE,
            expr.span,
            "an exclusive range would be more readable",
            |diag| {
                let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").maybe_paren().to_string());
                let end = Sugg::hir(cx, y, "y").maybe_paren();
                diag.span_suggestion(
                    expr.span,
                    "use",
                    format!("{start}..{end}"),
                    Applicability::MachineApplicable, // snippet
                );
            },
        );
    }
}

fn check_reversed_empty_range(cx: &LateContext<'_>, expr: &Expr<'_>) {
    fn inside_indexing_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
        matches!(
            get_parent_expr(cx, expr),
            Some(Expr {
                kind: ExprKind::Index(..),
                ..
            })
        )
    }

    fn is_for_loop_arg(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
        let mut cur_expr = expr;
        while let Some(parent_expr) = get_parent_expr(cx, cur_expr) {
            match higher::ForLoop::hir(parent_expr) {
                Some(higher::ForLoop { arg, .. }) if arg.hir_id == expr.hir_id => return true,
                _ => cur_expr = parent_expr,
            }
        }

        false
    }

    fn is_empty_range(limits: RangeLimits, ordering: Ordering) -> bool {
        match limits {
            RangeLimits::HalfOpen => ordering != Ordering::Less,
            RangeLimits::Closed => ordering == Ordering::Greater,
        }
    }

    if let Some(higher::Range {
        start: Some(start),
        end: Some(end),
        limits,
    }) = higher::Range::hir(expr)
        && let ty = cx.typeck_results().expr_ty(start)
        && let ty::Int(_) | ty::Uint(_) = ty.kind()
        && let ecx = ConstEvalCtxt::new(cx)
        && let Some(start_idx) = ecx.eval(start)
        && let Some(end_idx) = ecx.eval(end)
        && let Some(ordering) = Constant::partial_cmp(cx.tcx, ty, &start_idx, &end_idx)
        && is_empty_range(limits, ordering)
    {
        if inside_indexing_expr(cx, expr) {
            // Avoid linting `N..N` as it has proven to be useful, see #5689 and #5628 ...
            if ordering != Ordering::Equal {
                span_lint(
                    cx,
                    REVERSED_EMPTY_RANGES,
                    expr.span,
                    "this range is reversed and using it to index a slice will panic at run-time",
                );
            }
        // ... except in for loop arguments for backwards compatibility with `reverse_range_loop`
        } else if ordering != Ordering::Equal || is_for_loop_arg(cx, expr) {
            span_lint_and_then(
                cx,
                REVERSED_EMPTY_RANGES,
                expr.span,
                "this range is empty so it will yield no values",
                |diag| {
                    if ordering != Ordering::Equal {
                        let start_snippet = snippet(cx, start.span, "_");
                        let end_snippet = snippet(cx, end.span, "_");
                        let dots = match limits {
                            RangeLimits::HalfOpen => "..",
                            RangeLimits::Closed => "..=",
                        };

                        diag.span_suggestion(
                            expr.span,
                            "consider using the following if you are attempting to iterate over this \
                             range in reverse",
                            format!("({end_snippet}{dots}{start_snippet}).rev()"),
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
        }
    }
}

fn y_plus_one<'t>(cx: &LateContext<'_>, expr: &'t Expr<'_>) -> Option<&'t Expr<'t>> {
    match expr.kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            lhs,
            rhs,
        ) => {
            if is_integer_const(cx, lhs, 1) {
                Some(rhs)
            } else if is_integer_const(cx, rhs, 1) {
                Some(lhs)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn y_minus_one<'t>(cx: &LateContext<'_>, expr: &'t Expr<'_>) -> Option<&'t Expr<'t>> {
    match expr.kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Sub, ..
            },
            lhs,
            rhs,
        ) if is_integer_const(cx, rhs, 1) => Some(lhs),
        _ => None,
    }
}
