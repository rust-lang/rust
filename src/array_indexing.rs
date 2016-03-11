use rustc::lint::*;
use rustc::middle::const_eval::EvalHint::ExprTypeChecked;
use rustc::middle::const_eval::{eval_const_expr_partial, ConstVal};
use rustc::middle::ty::TyArray;
use rustc_front::hir::*;
use syntax::ast::RangeLimits;
use utils;

/// **What it does:** Check for out of bounds array indexing with a constant index.
///
/// **Why is this bad?** This will always panic at runtime.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
///
/// ```
/// let x = [1,2,3,4];
/// ...
/// x[9];
/// &x[2..9];
/// ```
declare_lint! {
    pub OUT_OF_BOUNDS_INDEXING,
    Deny,
    "out of bound constant indexing"
}

/// **What it does:** Check for usage of indexing or slicing.
///
/// **Why is this bad?** Usually, this can be safely allowed. However,
/// in some domains such as kernel development, a panic can cause the
/// whole operating system to crash.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
///
/// ```
/// ...
/// x[2];
/// &x[0..2];
/// ```
declare_lint! {
    pub INDEXING_SLICING,
    Allow,
    "indexing/slicing usage"
}

#[derive(Copy,Clone)]
pub struct ArrayIndexing;

impl LintPass for ArrayIndexing {
    fn get_lints(&self) -> LintArray {
        lint_array!(INDEXING_SLICING, OUT_OF_BOUNDS_INDEXING)
    }
}

impl LateLintPass for ArrayIndexing {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprIndex(ref array, ref index) = e.node {
            // Array with known size can be checked statically
            let ty = cx.tcx.expr_ty(array);
            if let TyArray(_, size) = ty.sty {
                let size = size as u64;

                // Index is a constant uint
                let const_index = eval_const_expr_partial(cx.tcx, &index, ExprTypeChecked, None);
                if let Ok(ConstVal::Uint(const_index)) = const_index {
                    if size <= const_index {
                        utils::span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "const index is out of bounds");
                        utils::span_lint(cx, INDEXING_SLICING, e.span, "indexing may panic");
                    } else {
                        // Index is within bounds
                        return;
                    }
                }

                // Index is a constant range
                if let Some(range) = utils::unsugar_range(index) {
                    let start = range.start.map(|start|
                        eval_const_expr_partial(cx.tcx, start, ExprTypeChecked, None));
                    let end = range.end.map(|end|
                        eval_const_expr_partial(cx.tcx, end, ExprTypeChecked, None));

                    if let Some((start, end)) = to_const_range(start, end, range.limits, size) {
                        if start >= size && end >= size {
                            utils::span_lint(cx,
                                             OUT_OF_BOUNDS_INDEXING,
                                             e.span,
                                             "range is out of bounds");
                            utils::span_lint(cx,
                                             INDEXING_SLICING,
                                             e.span,
                                             "slicing may panic");
                        } else {
                            // Range is within bounds
                            return;
                        }
                    }
                }
            }

            if let Some(range) = utils::unsugar_range(index) {
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

/// Returns an option containing a tuple with the start and end (exclusive) of the range
///
/// Note: we assume the start and the end of the range are unsigned, since array slicing
/// works only on usize
fn to_const_range<T>(start: Option<Result<ConstVal, T>>,
                    end: Option<Result<ConstVal, T>>,
                    limits: RangeLimits,
                    array_size: u64)
                    -> Option<(u64, u64)> {
    let start = match start {
        Some(Ok(ConstVal::Uint(x))) => x,
        Some(_) => return None,
        None => 0,
    };

    let end = match end {
        Some(Ok(ConstVal::Uint(x))) => {
            if limits == RangeLimits::Closed {
                x
            } else {
                x - 1
            }
        }
        Some(_) => return None,
        None => array_size - 1,
    };

    Some((start, end))
}
