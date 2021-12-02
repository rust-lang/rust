//! lint on indexing and slicing operations

use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::higher;
use rustc_ast::ast::RangeLimits;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for out of bounds array indexing with a constant
    /// index.
    ///
    /// ### Why is this bad?
    /// This will always panic at runtime.
    ///
    /// ### Known problems
    /// Hopefully none.
    ///
    /// ### Example
    /// ```no_run
    /// # #![allow(const_err)]
    /// let x = [1, 2, 3, 4];
    ///
    /// // Bad
    /// x[9];
    /// &x[2..9];
    ///
    /// // Good
    /// x[0];
    /// x[3];
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OUT_OF_BOUNDS_INDEXING,
    correctness,
    "out of bounds constant indexing"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of indexing or slicing. Arrays are special cases, this lint
    /// does report on arrays if we can tell that slicing operations are in bounds and does not
    /// lint on constant `usize` indexing on arrays because that is handled by rustc's `const_err` lint.
    ///
    /// ### Why is this bad?
    /// Indexing and slicing can panic at runtime and there are
    /// safe alternatives.
    ///
    /// ### Known problems
    /// Hopefully none.
    ///
    /// ### Example
    /// ```rust,no_run
    /// // Vector
    /// let x = vec![0; 5];
    ///
    /// // Bad
    /// x[2];
    /// &x[2..100];
    /// &x[2..];
    /// &x[..100];
    ///
    /// // Good
    /// x.get(2);
    /// x.get(2..100);
    /// x.get(2..);
    /// x.get(..100);
    ///
    /// // Array
    /// let y = [0, 1, 2, 3];
    ///
    /// // Bad
    /// &y[10..100];
    /// &y[10..];
    /// &y[..100];
    ///
    /// // Good
    /// &y[2..];
    /// &y[..2];
    /// &y[0..3];
    /// y.get(10);
    /// y.get(10..100);
    /// y.get(10..);
    /// y.get(..100);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INDEXING_SLICING,
    restriction,
    "indexing/slicing usage"
}

declare_lint_pass!(IndexingSlicing => [INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING]);

impl<'tcx> LateLintPass<'tcx> for IndexingSlicing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Index(array, index) = &expr.kind {
            let ty = cx.typeck_results().expr_ty(array).peel_refs();
            if let Some(range) = higher::Range::hir(index) {
                // Ranged indexes, i.e., &x[n..m], &x[n..], &x[..n] and &x[..]
                if let ty::Array(_, s) = ty.kind() {
                    let size: u128 = if let Some(size) = s.try_eval_usize(cx.tcx, cx.param_env) {
                        size.into()
                    } else {
                        return;
                    };

                    let const_range = to_const_range(cx, range, size);

                    if let (Some(start), _) = const_range {
                        if start > size {
                            span_lint(
                                cx,
                                OUT_OF_BOUNDS_INDEXING,
                                range.start.map_or(expr.span, |start| start.span),
                                "range is out of bounds",
                            );
                            return;
                        }
                    }

                    if let (_, Some(end)) = const_range {
                        if end > size {
                            span_lint(
                                cx,
                                OUT_OF_BOUNDS_INDEXING,
                                range.end.map_or(expr.span, |end| end.span),
                                "range is out of bounds",
                            );
                            return;
                        }
                    }

                    if let (Some(_), Some(_)) = const_range {
                        // early return because both start and end are constants
                        // and we have proven above that they are in bounds
                        return;
                    }
                }

                let help_msg = match (range.start, range.end) {
                    (None, Some(_)) => "consider using `.get(..n)`or `.get_mut(..n)` instead",
                    (Some(_), None) => "consider using `.get(n..)` or .get_mut(n..)` instead",
                    (Some(_), Some(_)) => "consider using `.get(n..m)` or `.get_mut(n..m)` instead",
                    (None, None) => return, // [..] is ok.
                };

                span_lint_and_help(cx, INDEXING_SLICING, expr.span, "slicing may panic", None, help_msg);
            } else {
                // Catchall non-range index, i.e., [n] or [n << m]
                if let ty::Array(..) = ty.kind() {
                    // Index is a constant uint.
                    if let Some(..) = constant(cx, cx.typeck_results(), index) {
                        // Let rustc's `const_err` lint handle constant `usize` indexing on arrays.
                        return;
                    }
                }

                span_lint_and_help(
                    cx,
                    INDEXING_SLICING,
                    expr.span,
                    "indexing may panic",
                    None,
                    "consider using `.get(n)` or `.get_mut(n)` instead",
                );
            }
        }
    }
}

/// Returns a tuple of options with the start and end (exclusive) values of
/// the range. If the start or end is not constant, None is returned.
fn to_const_range<'tcx>(
    cx: &LateContext<'tcx>,
    range: higher::Range<'_>,
    array_size: u128,
) -> (Option<u128>, Option<u128>) {
    let s = range
        .start
        .map(|expr| constant(cx, cx.typeck_results(), expr).map(|(c, _)| c));
    let start = match s {
        Some(Some(Constant::Int(x))) => Some(x),
        Some(_) => None,
        None => Some(0),
    };

    let e = range
        .end
        .map(|expr| constant(cx, cx.typeck_results(), expr).map(|(c, _)| c));
    let end = match e {
        Some(Some(Constant::Int(x))) => {
            if range.limits == RangeLimits::Closed {
                Some(x + 1)
            } else {
                Some(x)
            }
        },
        Some(_) => None,
        None => Some(array_size),
    };

    (start, end)
}
