use crate::utils::{
    contains_name, get_pat_name, match_type, paths, single_segment_path, snippet_with_applicability,
    span_lint_and_sugg, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc_ast::ast::UintTy;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Symbol;

declare_clippy_lint! {
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
    /// # let vec = vec![1_u8];
    /// &vec.iter().filter(|x| **x == 0u8).count(); // use bytecount::count instead
    /// ```
    pub NAIVE_BYTECOUNT,
    perf,
    "use of naive `<slice>.filter(|&x| x == y).count()` to count byte values"
}

declare_lint_pass!(ByteCount => [NAIVE_BYTECOUNT]);

impl<'tcx> LateLintPass<'tcx> for ByteCount {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(ref count, _, ref count_args, _) = expr.kind;
            if count.ident.name == sym!(count);
            if count_args.len() == 1;
            if let ExprKind::MethodCall(ref filter, _, ref filter_args, _) = count_args[0].kind;
            if filter.ident.name == sym!(filter);
            if filter_args.len() == 2;
            if let ExprKind::Closure(_, _, body_id, _, _) = filter_args[1].kind;
            then {
                let body = cx.tcx.hir().body(body_id);
                if_chain! {
                    if body.params.len() == 1;
                    if let Some(argname) = get_pat_name(&body.params[0].pat);
                    if let ExprKind::Binary(ref op, ref l, ref r) = body.value.kind;
                    if op.node == BinOpKind::Eq;
                    if match_type(cx,
                               walk_ptrs_ty(cx.typeck_results().expr_ty(&filter_args[0])),
                               &paths::SLICE_ITER);
                    then {
                        let needle = match get_path_name(l) {
                            Some(name) if check_arg(name, argname, r) => r,
                            _ => match get_path_name(r) {
                                Some(name) if check_arg(name, argname, l) => l,
                                _ => { return; }
                            }
                        };
                        if ty::Uint(UintTy::U8) != *walk_ptrs_ty(cx.typeck_results().expr_ty(needle)).kind() {
                            return;
                        }
                        let haystack = if let ExprKind::MethodCall(ref path, _, ref args, _) =
                                filter_args[0].kind {
                            let p = path.ident.name;
                            if (p == sym!(iter) || p == sym!(iter_mut)) && args.len() == 1 {
                                &args[0]
                            } else {
                                &filter_args[0]
                            }
                        } else {
                            &filter_args[0]
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
        };
    }
}

fn check_arg(name: Symbol, arg: Symbol, needle: &Expr<'_>) -> bool {
    name == arg && !contains_name(name, needle)
}

fn get_path_name(expr: &Expr<'_>) -> Option<Symbol> {
    match expr.kind {
        ExprKind::Box(ref e) | ExprKind::AddrOf(BorrowKind::Ref, _, ref e) | ExprKind::Unary(UnOp::UnDeref, ref e) => {
            get_path_name(e)
        },
        ExprKind::Block(ref b, _) => {
            if b.stmts.is_empty() {
                b.expr.as_ref().and_then(|p| get_path_name(p))
            } else {
                None
            }
        },
        ExprKind::Path(ref qpath) => single_segment_path(qpath).map(|ps| ps.ident.name),
        _ => None,
    }
}
