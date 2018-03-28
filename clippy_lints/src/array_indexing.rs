use rustc::lint::*;
use rustc::ty;
use rustc::hir;
use syntax::ast::RangeLimits;
use utils::{self, higher};
use consts::{constant, Constant};

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
                let size = size.val.to_raw_bits().unwrap();

                // Index is a constant uint
                if let Some((Constant::Int(const_index), _)) = constant(cx, index) {
                    if size <= const_index {
                        utils::span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "const index is out of bounds");
                    }

                    return;
                }

                // Index is a constant range
                if let Some(range) = higher::range(index) {
                    let start = range.start.map(|start| constant(cx, start).map(|(c, _)| c));
                    let end = range.end.map(|end| constant(cx, end).map(|(c, _)| c));

                    if let Some((start, end)) = to_const_range(&start, &end, range.limits, size) {
                        if start > size || end > size {
                            utils::span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "range is out of bounds");
                        }
                        return;
                    }
                }
            }

            if let Some(range) = higher::range(index) {
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
fn to_const_range(
    start: &Option<Option<Constant>>,
    end: &Option<Option<Constant>>,
    limits: RangeLimits,
    array_size: u128,
) -> Option<(u128, u128)> {
    let start = match *start {
        Some(Some(Constant::Int(x))) => x,
        Some(_) => return None,
        None => 0,
    };

    let end = match *end {
        Some(Some(Constant::Int(x))) => if limits == RangeLimits::Closed {
            x + 1
        } else {
            x
        },
        Some(_) => return None,
        None => array_size,
    };

    Some((start, end))
}
