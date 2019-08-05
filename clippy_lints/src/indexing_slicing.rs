//! lint on indexing and slicing operations

use crate::consts::{constant, Constant};
use crate::utils;
use crate::utils::higher;
use crate::utils::higher::Range;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::RangeLimits;

declare_clippy_lint! {
    /// **What it does:** Checks for out of bounds array indexing with a constant
    /// index.
    ///
    /// **Why is this bad?** This will always panic at runtime.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
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
    pub OUT_OF_BOUNDS_INDEXING,
    correctness,
    "out of bounds constant indexing"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of indexing or slicing. Arrays are special cases, this lint
    /// does report on arrays if we can tell that slicing operations are in bounds and does not
    /// lint on constant `usize` indexing on arrays because that is handled by rustc's `const_err` lint.
    ///
    /// **Why is this bad?** Indexing and slicing can panic at runtime and there are
    /// safe alternatives.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
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
    pub INDEXING_SLICING,
    restriction,
    "indexing/slicing usage"
}

declare_lint_pass!(IndexingSlicing => [INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for IndexingSlicing {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Index(ref array, ref index) = &expr.node {
            let ty = cx.tables.expr_ty(array);
            if let Some(range) = higher::range(cx, index) {
                // Ranged indexes, i.e., &x[n..m], &x[n..], &x[..n] and &x[..]
                if let ty::Array(_, s) = ty.sty {
                    let size: u128 = s.assert_usize(cx.tcx).unwrap().into();

                    let const_range = to_const_range(cx, range, size);

                    if let (Some(start), _) = const_range {
                        if start > size {
                            utils::span_lint(
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
                            utils::span_lint(
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
                    (None, Some(_)) => "Consider using `.get(..n)`or `.get_mut(..n)` instead",
                    (Some(_), None) => "Consider using `.get(n..)` or .get_mut(n..)` instead",
                    (Some(_), Some(_)) => "Consider using `.get(n..m)` or `.get_mut(n..m)` instead",
                    (None, None) => return, // [..] is ok.
                };

                utils::span_help_and_lint(cx, INDEXING_SLICING, expr.span, "slicing may panic.", help_msg);
            } else {
                // Catchall non-range index, i.e., [n] or [n << m]
                if let ty::Array(..) = ty.sty {
                    // Index is a constant uint.
                    if let Some(..) = constant(cx, cx.tables, index) {
                        // Let rustc's `const_err` lint handle constant `usize` indexing on arrays.
                        return;
                    }
                }

                utils::span_help_and_lint(
                    cx,
                    INDEXING_SLICING,
                    expr.span,
                    "indexing may panic.",
                    "Consider using `.get(n)` or `.get_mut(n)` instead",
                );
            }
        }
    }
}

/// Returns a tuple of options with the start and end (exclusive) values of
/// the range. If the start or end is not constant, None is returned.
fn to_const_range<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    range: Range<'_>,
    array_size: u128,
) -> (Option<u128>, Option<u128>) {
    let s = range.start.map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
    let start = match s {
        Some(Some(Constant::Int(x))) => Some(x),
        Some(_) => None,
        None => Some(0),
    };

    let e = range.end.map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
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
