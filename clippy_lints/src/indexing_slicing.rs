//! lint on indexing and slicing operations

use crate::consts::{constant, Constant};
use crate::utils::higher::Range;
use crate::utils::{self, higher};
use rustc::hir::*;
use rustc::lint::*;
use rustc::ty;
use syntax::ast::RangeLimits;

/// **What it does:** Checks for out of bounds array indexing with a constant
/// index.
///
/// **Why is this bad?** This will always panic at runtime.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
/// ```rust
/// let x = [1,2,3,4];
///
/// // Bad
/// x[9];
/// &x[2..9];
///
/// // Good
/// x[0];
/// x[3];
/// ```
declare_clippy_lint! {
    pub OUT_OF_BOUNDS_INDEXING,
    correctness,
    "out of bounds constant indexing"
}

/// **What it does:** Checks for usage of indexing or slicing. Does not report
/// if we can tell that the indexing or slicing operations on an array are in
/// bounds.
///
/// **Why is this bad?** Indexing and slicing can panic at runtime and there are
/// safe alternatives.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
/// ```rust
/// let x = vec![0; 5];
/// // Bad
/// x[2];
/// &x[2..100];
/// &x[2..];
/// &x[..100];
///
/// // Good
/// x.get(2)
/// x.get(2..100)
/// x.get(2..)
/// x.get(..100)
/// ```
declare_clippy_lint! {
    pub INDEXING_SLICING,
    restriction,
    "indexing/slicing usage"
}

#[derive(Copy, Clone)]
pub struct IndexingSlicingPass;

impl LintPass for IndexingSlicingPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for IndexingSlicingPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprIndex(ref a, ref b) = &expr.node {
            match &b.node {
                // Both ExprStruct and ExprPath require this approach's checks
                // on the `range` returned by `higher::range(cx, b)`.
                // ExprStruct handles &x[n..m], &x[n..] and &x[..n].
                // ExprPath handles &x[..] and x[var]
                ExprStruct(_, _, _) | ExprPath(_) => {
                    if let Some(range) = higher::range(cx, b) {
                        let ty = cx.tables.expr_ty(a);
                        if let ty::TyArray(_, s) = ty.sty {
                            let size: u128 = s.assert_usize(cx.tcx).unwrap().into();
                            // Index is a constant range.
                            if let Some((start, end)) = to_const_range(cx, range, size) {
                                if start > size || end > size {
                                    utils::span_lint(
                                        cx,
                                        OUT_OF_BOUNDS_INDEXING,
                                        expr.span,
                                        "range is out of bounds",
                                    );
                                } else {
                                    // Range is in bounds, ok.
                                    return;
                                }
                            }
                        }
                        match (range.start, range.end) {
                            (None, Some(_)) => {
                                cx.span_lint(
                                    INDEXING_SLICING,
                                    expr.span,
                                    "slicing may panic. Consider using \
                                     `.get(..n)`or `.get_mut(..n)` instead",
                                );
                            }
                            (Some(_), None) => {
                                cx.span_lint(
                                    INDEXING_SLICING,
                                    expr.span,
                                    "slicing may panic. Consider using \
                                     `.get(n..)` or .get_mut(n..)` instead",
                                );
                            }
                            (Some(_), Some(_)) => {
                                cx.span_lint(
                                    INDEXING_SLICING,
                                    expr.span,
                                    "slicing may panic. Consider using \
                                     `.get(n..m)` or `.get_mut(n..m)` instead",
                                );
                            }
                            (None, None) => (),
                        }
                    } else {
                        cx.span_lint(
                            INDEXING_SLICING,
                            expr.span,
                            "indexing may panic. Consider using `.get(n)` or \
                             `.get_mut(n)` instead",
                        );
                    }
                }
                ExprLit(_) => {
                    // [n]
                    let ty = cx.tables.expr_ty(a);
                    if let ty::TyArray(_, s) = ty.sty {
                        let size: u128 = s.assert_usize(cx.tcx).unwrap().into();
                        // Index is a constant uint.
                        if let Some((Constant::Int(const_index), _)) = constant(cx, cx.tables, b) {
                            if size <= const_index {
                                utils::span_lint(
                                    cx,
                                    OUT_OF_BOUNDS_INDEXING,
                                    expr.span,
                                    "const index is out of bounds",
                                );
                            }
                            // Else index is in bounds, ok.
                        }
                    } else {
                        cx.span_lint(
                            INDEXING_SLICING,
                            expr.span,
                            "indexing may panic. Consider using `.get(n)` or \
                             `.get_mut(n)` instead",
                        );
                    }
                }
                _ => (),
            }
        }
    }
}

/// Returns an option containing a tuple with the start and end (exclusive) of
/// the range.
fn to_const_range<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    range: Range,
    array_size: u128,
) -> Option<(u128, u128)> {
    let s = range
        .start
        .map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
    let start = match s {
        Some(Some(Constant::Int(x))) => x,
        Some(_) => return None,
        None => 0,
    };

    let e = range
        .end
        .map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
    let end = match e {
        Some(Some(Constant::Int(x))) => if range.limits == RangeLimits::Closed {
            x + 1
        } else {
            x
        },
        Some(_) => return None,
        None => array_size,
    };

    Some((start, end))
}
