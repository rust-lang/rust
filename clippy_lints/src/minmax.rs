use clippy_utils::consts::{constant_simple, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{match_def_path, match_trait_method, paths};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::cmp::Ordering;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions where `std::cmp::min` and `max` are
    /// used to clamp values, but switched so that the result is constant.
    ///
    /// ### Why is this bad?
    /// This is in all probability not the intended outcome. At
    /// the least it hurts readability of the code.
    ///
    /// ### Example
    /// ```ignore
    /// min(0, max(100, x))
    /// ```
    /// or
    /// ```ignore
    /// x.max(100).min(0)
    /// ```
    /// It will always be equal to `0`. Probably the author meant to clamp the value
    /// between 0 and 100, but has erroneously swapped `min` and `max`.
    #[clippy::version = "pre 1.29.0"]
    pub MIN_MAX,
    correctness,
    "`min(_, max(_, _))` (or vice versa) with bounds clamping the result to a constant"
}

declare_lint_pass!(MinMaxPass => [MIN_MAX]);

impl<'tcx> LateLintPass<'tcx> for MinMaxPass {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some((outer_max, outer_c, oe)) = min_max(cx, expr) {
            if let Some((inner_max, inner_c, ie)) = min_max(cx, oe) {
                if outer_max == inner_max {
                    return;
                }
                match (
                    outer_max,
                    Constant::partial_cmp(cx.tcx, cx.typeck_results().expr_ty(ie), &outer_c, &inner_c),
                ) {
                    (_, None) | (MinMax::Max, Some(Ordering::Less)) | (MinMax::Min, Some(Ordering::Greater)) => (),
                    _ => {
                        span_lint(
                            cx,
                            MIN_MAX,
                            expr.span,
                            "this `min`/`max` combination leads to constant result",
                        );
                    },
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum MinMax {
    Min,
    Max,
}

fn min_max<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<(MinMax, Constant, &'a Expr<'a>)> {
    match expr.kind {
        ExprKind::Call(path, args) => {
            if let ExprKind::Path(ref qpath) = path.kind {
                cx.typeck_results()
                    .qpath_res(qpath, path.hir_id)
                    .opt_def_id()
                    .and_then(|def_id| {
                        if match_def_path(cx, def_id, &paths::CMP_MIN) {
                            fetch_const(cx, args, MinMax::Min)
                        } else if match_def_path(cx, def_id, &paths::CMP_MAX) {
                            fetch_const(cx, args, MinMax::Max)
                        } else {
                            None
                        }
                    })
            } else {
                None
            }
        },
        ExprKind::MethodCall(path, _, args, _) => {
            if_chain! {
                if let [obj, _] = args;
                if cx.typeck_results().expr_ty(obj).is_floating_point() || match_trait_method(cx, expr, &paths::ORD);
                then {
                    if path.ident.name == sym!(max) {
                        fetch_const(cx, args, MinMax::Max)
                    } else if path.ident.name == sym!(min) {
                        fetch_const(cx, args, MinMax::Min)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        },
        _ => None,
    }
}

fn fetch_const<'a>(cx: &LateContext<'_>, args: &'a [Expr<'a>], m: MinMax) -> Option<(MinMax, Constant, &'a Expr<'a>)> {
    if args.len() != 2 {
        return None;
    }
    constant_simple(cx, cx.typeck_results(), &args[0]).map_or_else(
        || constant_simple(cx, cx.typeck_results(), &args[1]).map(|c| (m, c, &args[0])),
        |c| {
            if constant_simple(cx, cx.typeck_results(), &args[1]).is_none() {
                // otherwise ignore
                Some((m, c, &args[1]))
            } else {
                None
            }
        },
    )
}
