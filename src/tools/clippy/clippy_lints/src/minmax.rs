use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::sym;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::SyntaxContext;
use std::cmp::Ordering::{Equal, Greater, Less};

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
    /// ```rust,ignore
    /// min(0, max(100, x))
    ///
    /// // or
    ///
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
        if let Some((outer_max, outer_c, oe)) = min_max(cx, expr)
            && let Some((inner_max, inner_c, ie)) = min_max(cx, oe)
            && outer_max != inner_max
            && let Some(ord) = Constant::partial_cmp(cx.tcx, cx.typeck_results().expr_ty(ie), &outer_c, &inner_c)
            && matches!(
                (outer_max, ord),
                (MinMax::Max, Equal | Greater) | (MinMax::Min, Equal | Less)
            )
        {
            span_lint(
                cx,
                MIN_MAX,
                expr.span,
                "this `min`/`max` combination leads to constant result",
            );
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
                    .and_then(|def_id| match cx.tcx.get_diagnostic_name(def_id) {
                        Some(sym::cmp_min) => fetch_const(cx, expr.span.ctxt(), None, args, MinMax::Min),
                        Some(sym::cmp_max) => fetch_const(cx, expr.span.ctxt(), None, args, MinMax::Max),
                        _ => None,
                    })
            } else {
                None
            }
        },
        ExprKind::MethodCall(path, receiver, args @ [_], _) => {
            if cx.typeck_results().expr_ty(receiver).is_floating_point()
                || cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Ord)
            {
                match path.ident.name {
                    sym::max => fetch_const(cx, expr.span.ctxt(), Some(receiver), args, MinMax::Max),
                    sym::min => fetch_const(cx, expr.span.ctxt(), Some(receiver), args, MinMax::Min),
                    _ => None,
                }
            } else {
                None
            }
        },
        _ => None,
    }
}

fn fetch_const<'a>(
    cx: &LateContext<'_>,
    ctxt: SyntaxContext,
    receiver: Option<&'a Expr<'a>>,
    args: &'a [Expr<'a>],
    m: MinMax,
) -> Option<(MinMax, Constant, &'a Expr<'a>)> {
    let mut args = receiver.into_iter().chain(args);
    let first_arg = args.next()?;
    let second_arg = args.next()?;
    if args.next().is_some() {
        return None;
    }
    let ecx = ConstEvalCtxt::new(cx);
    match (ecx.eval_local(first_arg, ctxt), ecx.eval_local(second_arg, ctxt)) {
        (Some(c), None) => Some((m, c, second_arg)),
        (None, Some(c)) => Some((m, c, first_arg)),
        // otherwise ignore
        _ => None,
    }
}
