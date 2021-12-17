use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::match_type;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{path_to_local_id, paths, peel_blocks, peel_ref_operators, strip_pat_refs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, UintTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for naive byte counts
    ///
    /// ### Why is this bad?
    /// The [`bytecount`](https://crates.io/crates/bytecount)
    /// crate has methods to count your bytes faster, especially for large slices.
    ///
    /// ### Known problems
    /// If you have predominantly small slices, the
    /// `bytecount::count(..)` method may actually be slower. However, if you can
    /// ensure that less than 2³²-1 matches arise, the `naive_count_32(..)` can be
    /// faster in those cases.
    ///
    /// ### Example
    /// ```rust
    /// # let vec = vec![1_u8];
    /// &vec.iter().filter(|x| **x == 0u8).count(); // use bytecount::count instead
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NAIVE_BYTECOUNT,
    pedantic,
    "use of naive `<slice>.filter(|&x| x == y).count()` to count byte values"
}

declare_lint_pass!(ByteCount => [NAIVE_BYTECOUNT]);

impl<'tcx> LateLintPass<'tcx> for ByteCount {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(count, _, [count_recv], _) = expr.kind;
            if count.ident.name == sym::count;
            if let ExprKind::MethodCall(filter, _, [filter_recv, filter_arg], _) = count_recv.kind;
            if filter.ident.name == sym!(filter);
            if let ExprKind::Closure(_, _, body_id, _, _) = filter_arg.kind;
            let body = cx.tcx.hir().body(body_id);
            if let [param] = body.params;
            if let PatKind::Binding(_, arg_id, _, _) = strip_pat_refs(param.pat).kind;
            if let ExprKind::Binary(ref op, l, r) = body.value.kind;
            if op.node == BinOpKind::Eq;
            if match_type(cx,
                       cx.typeck_results().expr_ty(filter_recv).peel_refs(),
                       &paths::SLICE_ITER);
            let operand_is_arg = |expr| {
                let expr = peel_ref_operators(cx, peel_blocks(expr));
                path_to_local_id(expr, arg_id)
            };
            let needle = if operand_is_arg(l) {
                r
            } else if operand_is_arg(r) {
                l
            } else {
                return;
            };
            if ty::Uint(UintTy::U8) == *cx.typeck_results().expr_ty(needle).peel_refs().kind();
            if !is_local_used(cx, needle, arg_id);
            then {
                let haystack = if let ExprKind::MethodCall(path, _, args, _) =
                        filter_recv.kind {
                    let p = path.ident.name;
                    if (p == sym::iter || p == sym!(iter_mut)) && args.len() == 1 {
                        &args[0]
                    } else {
                        filter_recv
                    }
                } else {
                    filter_recv
                };
                let mut applicability = Applicability::MaybeIncorrect;
                span_lint_and_sugg(
                    cx,
                    NAIVE_BYTECOUNT,
                    expr.span,
                    "you appear to be counting bytes the naive way",
                    "consider using the bytecount crate",
                    format!("bytecount::count({}, {})",
                            snippet_with_applicability(cx, haystack.span, "..", &mut applicability),
                            snippet_with_applicability(cx, needle.span, "..", &mut applicability)),
                    applicability,
                );
            }
        };
    }
}
