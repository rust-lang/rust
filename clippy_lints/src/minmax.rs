use consts::{constant_simple, Constant};
use rustc::lint::*;
use rustc::hir::*;
use std::cmp::{Ordering, PartialOrd};
use utils::{match_def_path, paths, span_lint};

/// **What it does:** Checks for expressions where `std::cmp::min` and `max` are
/// used to clamp values, but switched so that the result is constant.
///
/// **Why is this bad?** This is in all probability not the intended outcome. At
/// the least it hurts readability of the code.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// min(0, max(100, x))
/// ```
/// It will always be equal to `0`. Probably the author meant to clamp the value
/// between 0 and 100, but has erroneously swapped `min` and `max`.
declare_lint! {
    pub MIN_MAX,
    Warn,
    "`min(_, max(_, _))` (or vice versa) with bounds clamping the result to a constant"
}

#[allow(missing_copy_implementations)]
pub struct MinMaxPass;

impl LintPass for MinMaxPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MIN_MAX)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MinMaxPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some((outer_max, outer_c, oe)) = min_max(cx, expr) {
            if let Some((inner_max, inner_c, _)) = min_max(cx, oe) {
                if outer_max == inner_max {
                    return;
                }
                match (outer_max, outer_c.partial_cmp(&inner_c)) {
                    (_, None) | (MinMax::Max, Some(Ordering::Less)) | (MinMax::Min, Some(Ordering::Greater)) => (),
                    _ => {
                        span_lint(cx, MIN_MAX, expr.span, "this min/max combination leads to constant result");
                    },
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum MinMax {
    Min,
    Max,
}

fn min_max<'a>(cx: &LateContext, expr: &'a Expr) -> Option<(MinMax, Constant, &'a Expr)> {
    if let ExprCall(ref path, ref args) = expr.node {
        if let ExprPath(ref qpath) = path.node {
            let def_id = cx.tables.qpath_def(qpath, path.hir_id).def_id();

            if match_def_path(cx.tcx, def_id, &paths::CMP_MIN) {
                fetch_const(cx, args, MinMax::Min)
            } else if match_def_path(cx.tcx, def_id, &paths::CMP_MAX) {
                fetch_const(cx, args, MinMax::Max)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

fn fetch_const<'a>(cx: &LateContext, args: &'a [Expr], m: MinMax) -> Option<(MinMax, Constant, &'a Expr)> {
    if args.len() != 2 {
        return None;
    }
    if let Some(c) = constant_simple(cx, &args[0]) {
        if constant_simple(cx, &args[1]).is_none() {
            // otherwise ignore
            Some((m, c, &args[1]))
        } else {
            None
        }
    } else if let Some(c) = constant_simple(cx, &args[1]) {
        Some((m, c, &args[0]))
    } else {
        None
    }
}
