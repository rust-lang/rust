use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, in_constant, is_integer_const, meets_msrv, msrvs, path_to_local};
use clippy_utils::{higher, SpanlessEq};
use if_chain::if_chain;
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, HirId, PathSegment, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{Span, Spanned};
use rustc_span::sym;
use std::cmp::Ordering;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for zipping a collection with the range of
    /// `0.._.len()`.
    ///
    /// ### Why is this bad?
    /// The code is better expressed with `.enumerate()`.
    ///
    /// ### Example
    /// ```rust
    /// # let x = vec![1];
    /// x.iter().zip(0..x.len());
    /// ```
    /// Could be written as
    /// ```rust
    /// # let x = vec![1];
    /// x.iter().enumerate();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_ZIP_WITH_LEN,
    complexity,
    "zipping iterator with a range when `enumerate()` would do"
}

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
    /// ```rust,ignore
    /// for x..(y+1) { .. }
    /// ```
    /// Could be written as
    /// ```rust,ignore
    /// for x..=y { .. }
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
    /// ```rust,ignore
    /// for x..=(y-1) { .. }
    /// ```
    /// Could be written as
    /// ```rust,ignore
    /// for x..y { .. }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub RANGE_MINUS_ONE,
    pedantic,
    "`x..=(y-1)` reads better as `x..y`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for range expressions `x..y` where both `x` and `y`
    /// are constant and `x` is greater or equal to `y`.
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
    /// ```rust
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
    /// ```rust
    /// // given
    /// let x = 6;
    ///
    /// assert!(x >= 3 && x < 8);
    /// ```
    /// Use instead:
    /// ```rust
    ///# let x = 6;
    /// assert!((3..8).contains(&x));
    /// ```
    #[clippy::version = "1.49.0"]
    pub MANUAL_RANGE_CONTAINS,
    style,
    "manually reimplementing {`Range`, `RangeInclusive`}`::contains`"
}

pub struct Ranges {
    msrv: Option<RustcVersion>,
}

impl Ranges {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(Ranges => [
    RANGE_ZIP_WITH_LEN,
    RANGE_PLUS_ONE,
    RANGE_MINUS_ONE,
    REVERSED_EMPTY_RANGES,
    MANUAL_RANGE_CONTAINS,
]);

impl<'tcx> LateLintPass<'tcx> for Ranges {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        match expr.kind {
            ExprKind::MethodCall(path, args, _) => {
                check_range_zip_with_len(cx, path, args, expr.span);
            },
            ExprKind::Binary(ref op, l, r) => {
                if meets_msrv(self.msrv.as_ref(), &msrvs::RANGE_CONTAINS) {
                    check_possible_range_contains(cx, op.node, l, r, expr);
                }
            },
            _ => {},
        }

        check_exclusive_range_plus_one(cx, expr);
        check_inclusive_range_minus_one(cx, expr);
        check_reversed_empty_range(cx, expr);
    }
    extract_msrv_attr!(LateContext);
}

fn check_possible_range_contains(cx: &LateContext<'_>, op: BinOpKind, l: &Expr<'_>, r: &Expr<'_>, expr: &Expr<'_>) {
    if in_constant(cx, expr.hir_id) {
        return;
    }

    let span = expr.span;
    let combine_and = match op {
        BinOpKind::And | BinOpKind::BitAnd => true,
        BinOpKind::Or | BinOpKind::BitOr => false,
        _ => return,
    };
    // value, name, order (higher/lower), inclusiveness
    if let (Some((lval, lid, name_span, lval_span, lord, linc)), Some((rval, rid, _, rval_span, rord, rinc))) =
        (check_range_bounds(cx, l), check_range_bounds(cx, r))
    {
        // we only lint comparisons on the same name and with different
        // direction
        if lid != rid || lord == rord {
            return;
        }
        let ord = Constant::partial_cmp(cx.tcx, cx.typeck_results().expr_ty(l), &lval, &rval);
        if combine_and && ord == Some(rord) {
            // order lower bound and upper bound
            let (l_span, u_span, l_inc, u_inc) = if rord == Ordering::Less {
                (lval_span, rval_span, linc, rinc)
            } else {
                (rval_span, lval_span, rinc, linc)
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
            let name = snippet_with_applicability(cx, name_span, "_", &mut applicability);
            let lo = snippet_with_applicability(cx, l_span, "_", &mut applicability);
            let hi = snippet_with_applicability(cx, u_span, "_", &mut applicability);
            let space = if lo.ends_with('.') { " " } else { "" };
            span_lint_and_sugg(
                cx,
                MANUAL_RANGE_CONTAINS,
                span,
                &format!("manual `{}::contains` implementation", range_type),
                "use",
                format!("({}{}{}{}).contains(&{})", lo, space, range_op, hi, name),
                applicability,
            );
        } else if !combine_and && ord == Some(lord) {
            // `!_.contains(_)`
            // order lower bound and upper bound
            let (l_span, u_span, l_inc, u_inc) = if lord == Ordering::Less {
                (lval_span, rval_span, linc, rinc)
            } else {
                (rval_span, lval_span, rinc, linc)
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
            let name = snippet_with_applicability(cx, name_span, "_", &mut applicability);
            let lo = snippet_with_applicability(cx, l_span, "_", &mut applicability);
            let hi = snippet_with_applicability(cx, u_span, "_", &mut applicability);
            let space = if lo.ends_with('.') { " " } else { "" };
            span_lint_and_sugg(
                cx,
                MANUAL_RANGE_CONTAINS,
                span,
                &format!("manual `!{}::contains` implementation", range_type),
                "use",
                format!("!({}{}{}{}).contains(&{})", lo, space, range_op, hi, name),
                applicability,
            );
        }
    }
}

fn check_range_bounds(cx: &LateContext<'_>, ex: &Expr<'_>) -> Option<(Constant, HirId, Span, Span, Ordering, bool)> {
    if let ExprKind::Binary(ref op, l, r) = ex.kind {
        let (inclusive, ordering) = match op.node {
            BinOpKind::Gt => (false, Ordering::Greater),
            BinOpKind::Ge => (true, Ordering::Greater),
            BinOpKind::Lt => (false, Ordering::Less),
            BinOpKind::Le => (true, Ordering::Less),
            _ => return None,
        };
        if let Some(id) = path_to_local(l) {
            if let Some((c, _)) = constant(cx, cx.typeck_results(), r) {
                return Some((c, id, l.span, r.span, ordering, inclusive));
            }
        } else if let Some(id) = path_to_local(r) {
            if let Some((c, _)) = constant(cx, cx.typeck_results(), l) {
                return Some((c, id, r.span, l.span, ordering.reverse(), inclusive));
            }
        }
    }
    None
}

fn check_range_zip_with_len(cx: &LateContext<'_>, path: &PathSegment<'_>, args: &[Expr<'_>], span: Span) {
    if_chain! {
        if path.ident.as_str() == "zip";
        if let [iter, zip_arg] = args;
        // `.iter()` call
        if let ExprKind::MethodCall(iter_path, iter_args, _) = iter.kind;
        if iter_path.ident.name == sym::iter;
        // range expression in `.zip()` call: `0..x.len()`
        if let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::Range::hir(zip_arg);
        if is_integer_const(cx, start, 0);
        // `.len()` call
        if let ExprKind::MethodCall(len_path, len_args, _) = end.kind;
        if len_path.ident.name == sym::len && len_args.len() == 1;
        // `.iter()` and `.len()` called on same `Path`
        if let ExprKind::Path(QPath::Resolved(_, iter_path)) = iter_args[0].kind;
        if let ExprKind::Path(QPath::Resolved(_, len_path)) = len_args[0].kind;
        if SpanlessEq::new(cx).eq_path_segments(&iter_path.segments, &len_path.segments);
        then {
            span_lint(cx,
                RANGE_ZIP_WITH_LEN,
                span,
                &format!("it is more idiomatic to use `{}.iter().enumerate()`",
                    snippet(cx, iter_args[0].span, "_"))
            );
        }
    }
}

// exclusive range plus one: `x..(y+1)`
fn check_exclusive_range_plus_one(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let Some(higher::Range {
            start,
            end: Some(end),
            limits: RangeLimits::HalfOpen
        }) = higher::Range::hir(expr);
        if let Some(y) = y_plus_one(cx, end);
        then {
            let span = if expr.span.from_expansion() {
                expr.span
                    .ctxt()
                    .outer_expn_data()
                    .call_site
            } else {
                expr.span
            };
            span_lint_and_then(
                cx,
                RANGE_PLUS_ONE,
                span,
                "an inclusive range would be more readable",
                |diag| {
                    let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").maybe_par().to_string());
                    let end = Sugg::hir(cx, y, "y").maybe_par();
                    if let Some(is_wrapped) = &snippet_opt(cx, span) {
                        if is_wrapped.starts_with('(') && is_wrapped.ends_with(')') {
                            diag.span_suggestion(
                                span,
                                "use",
                                format!("({}..={})", start, end),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            diag.span_suggestion(
                                span,
                                "use",
                                format!("{}..={}", start, end),
                                Applicability::MachineApplicable, // snippet
                            );
                        }
                    }
                },
            );
        }
    }
}

// inclusive range minus one: `x..=(y-1)`
fn check_inclusive_range_minus_one(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let Some(higher::Range { start, end: Some(end), limits: RangeLimits::Closed }) = higher::Range::hir(expr);
        if let Some(y) = y_minus_one(cx, end);
        then {
            span_lint_and_then(
                cx,
                RANGE_MINUS_ONE,
                expr.span,
                "an exclusive range would be more readable",
                |diag| {
                    let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").maybe_par().to_string());
                    let end = Sugg::hir(cx, y, "y").maybe_par();
                    diag.span_suggestion(
                        expr.span,
                        "use",
                        format!("{}..{}", start, end),
                        Applicability::MachineApplicable, // snippet
                    );
                },
            );
        }
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

    if_chain! {
        if let Some(higher::Range { start: Some(start), end: Some(end), limits }) = higher::Range::hir(expr);
        let ty = cx.typeck_results().expr_ty(start);
        if let ty::Int(_) | ty::Uint(_) = ty.kind();
        if let Some((start_idx, _)) = constant(cx, cx.typeck_results(), start);
        if let Some((end_idx, _)) = constant(cx, cx.typeck_results(), end);
        if let Some(ordering) = Constant::partial_cmp(cx.tcx, ty, &start_idx, &end_idx);
        if is_empty_range(limits, ordering);
        then {
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
                                RangeLimits::Closed => "..="
                            };

                            diag.span_suggestion(
                                expr.span,
                                "consider using the following if you are attempting to iterate over this \
                                 range in reverse",
                                format!("({}{}{}).rev()", end_snippet, dots, start_snippet),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    },
                );
            }
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
