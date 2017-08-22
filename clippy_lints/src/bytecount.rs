use consts::{constant, Constant};
use rustc_const_math::ConstInt;
use rustc::hir::*;
use rustc::lint::*;
use utils::{match_type, paths, snippet, span_lint_and_sugg, walk_ptrs_ty};

/// **What it does:** Checks for naive byte counts
///
/// **Why is this bad?** The [`bytecount`](https://crates.io/crates/bytecount)
/// crate has methods to count your bytes faster, especially for large slices.
///
/// **Known problems:** If you have predominantly small slices, the
/// `bytecount::count(..)` method may actually be slower. However, if you can
/// ensure that less than 2³²-1 matches arise, the `naive_count_32(..)` can be
/// faster in those cases.
///
/// **Example:**
///
/// ```rust
/// &my_data.filter(|&x| x == 0u8).count() // use bytecount::count instead
/// ```
declare_lint! {
    pub NAIVE_BYTECOUNT,
    Warn,
    "use of naive `<slice>.filter(|&x| x == y).count()` to count byte values"
}

#[derive(Copy, Clone)]
pub struct ByteCount;

impl LintPass for ByteCount {
    fn get_lints(&self) -> LintArray {
        lint_array!(NAIVE_BYTECOUNT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ByteCount {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain!([
            let ExprMethodCall(ref count, _, ref count_args) = expr.node,
            count.name == "count",
            count_args.len() == 1,
            let ExprMethodCall(ref filter, _, ref filter_args) = count_args[0].node,
            filter.name == "filter",
            filter_args.len() == 2,
            let ExprClosure(_, _, body_id, _) = filter_args[1].node,
        ], {
            let body = cx.tcx.hir.body(body_id);
            if_let_chain!([
                let ExprBinary(ref op, ref l, ref r) = body.value.node,
                op.node == BiEq,
                match_type(cx,
                           walk_ptrs_ty(cx.tables.expr_ty(&filter_args[0])),
                           &paths::SLICE_ITER),
                let Some((Constant::Int(ConstInt::U8(needle)), _)) =
                        constant(cx, l).or_else(|| constant(cx, r))
            ], {
                let haystack = if let ExprMethodCall(ref path, _, ref args) =
                        filter_args[0].node {
                    let p = path.name;
                    if (p == "iter" || p == "iter_mut") && args.len() == 1 {
                        &args[0]
                    } else {
                        &filter_args[0]
                    }
                } else {
                    &filter_args[0]
                };
                span_lint_and_sugg(cx,
                                   NAIVE_BYTECOUNT,
                                   expr.span,
                                   "You appear to be counting bytes the naive way",
                                   "Consider using the bytecount crate",
                                   format!("bytecount::count({}, {})",
                                            snippet(cx, haystack.span, ".."),
                                            needle));
            });
        });
    }
}
