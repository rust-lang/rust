use crate::consts::{constant, Constant};
use crate::utils::higher::Range;
use crate::utils::{self, higher};
use rustc::hir;
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
/// ...
/// x[9];
/// &x[2..9];
/// ```
declare_clippy_lint! {
    pub OUT_OF_BOUNDS_INDEXING,
    correctness,
    "out of bounds constant indexing"
}

/// **What it does:** Checks for usage of indexing or slicing.
///
/// **Why is this bad?** Usually, this can be safely allowed. However, in some
/// domains such as kernel development, a panic can cause the whole operating
/// system to crash.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
/// ```rust
/// ...
/// x[2];
/// &x[0..2];
/// ```
declare_clippy_lint! {
    pub INDEXING_SLICING,
    restriction,
    "indexing/slicing usage"
}

#[derive(Copy, Clone)]
pub struct ArrayIndexing;

impl LintPass for ArrayIndexing {
    fn get_lints(&self) -> LintArray {
        lint_array!(INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ArrayIndexing {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx hir::Expr) {
        if let hir::ExprIndex(ref array, ref index) = e.node {
            // Array with known size can be checked statically
            let ty = cx.tables.expr_ty(array);
            if let ty::TyArray(_, size) = ty.sty {
                let size = size.assert_usize(cx.tcx).unwrap().into();

                // Index is a constant uint
                if let Some((Constant::Int(const_index), _)) = constant(cx, cx.tables, index) {
                    if size <= const_index {
                        utils::span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "const index is out of bounds");
                    }

                    return;
                }

                // Index is a constant range
                if let Some(range) = higher::range(cx, index) {
                    if let Some((start, end)) = to_const_range(cx, range, size) {
                        if start > size || end > size {
                            utils::span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "range is out of bounds");
                        }
                        return;
                    }
                }
            }

            if let Some(range) = higher::range(cx, index) {
                // Full ranges are always valid
                if range.start.is_none() && range.end.is_none() {
                    return;
                }

                // Impossible to know if indexing or slicing is correct
                utils::span_lint(cx, INDEXING_SLICING, e.span, "slicing may panic");
            } else {
                utils::span_lint(cx, INDEXING_SLICING, e.span, "indexing may panic");
            }
        }
    }
}

/// Returns an option containing a tuple with the start and end (exclusive) of
/// the range.
fn to_const_range<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, range: Range, array_size: u128) -> Option<(u128, u128)> {
    let s = range.start.map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
    let start = match s {
        Some(Some(Constant::Int(x))) => x,
        Some(_) => return None,
        None => 0,
    };

    let e = range.end.map(|expr| constant(cx, cx.tables, expr).map(|(c, _)| c));
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
