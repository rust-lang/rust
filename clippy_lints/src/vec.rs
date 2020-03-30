use crate::consts::constant;
use crate::utils::{higher, is_copy, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `&vec![..]` when using `&[..]` would
    /// be possible.
    ///
    /// **Why is this bad?** This is less efficient.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// foo(&vec![1, 2])
    /// ```
    pub USELESS_VEC,
    perf,
    "useless `vec!`"
}

declare_lint_pass!(UselessVec => [USELESS_VEC]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UselessVec {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        // search for `&vec![_]` expressions where the adjusted type is `&[_]`
        if_chain! {
            if let ty::Ref(_, ty, _) = cx.tables.expr_ty_adjusted(expr).kind;
            if let ty::Slice(..) = ty.kind;
            if let ExprKind::AddrOf(BorrowKind::Ref, _, ref addressee) = expr.kind;
            if let Some(vec_args) = higher::vec_macro(cx, addressee);
            then {
                check_vec_macro(cx, &vec_args, expr.span);
            }
        }

        // search for `for _ in vec![…]`
        if_chain! {
            if let Some((_, arg, _)) = higher::for_loop(expr);
            if let Some(vec_args) = higher::vec_macro(cx, arg);
            if is_copy(cx, vec_type(cx.tables.expr_ty_adjusted(arg)));
            then {
                // report the error around the `vec!` not inside `<std macros>:`
                let span = arg.span
                    .ctxt()
                    .outer_expn_data()
                    .call_site
                    .ctxt()
                    .outer_expn_data()
                    .call_site;
                check_vec_macro(cx, &vec_args, span);
            }
        }
    }
}

fn check_vec_macro<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, vec_args: &higher::VecArgs<'tcx>, span: Span) {
    let mut applicability = Applicability::MachineApplicable;
    let snippet = match *vec_args {
        higher::VecArgs::Repeat(elem, len) => {
            if constant(cx, cx.tables, len).is_some() {
                format!(
                    "&[{}; {}]",
                    snippet_with_applicability(cx, elem.span, "elem", &mut applicability),
                    snippet_with_applicability(cx, len.span, "len", &mut applicability)
                )
            } else {
                return;
            }
        },
        higher::VecArgs::Vec(args) => {
            if let Some(last) = args.iter().last() {
                let span = args[0].span.to(last.span);

                format!("&[{}]", snippet_with_applicability(cx, span, "..", &mut applicability))
            } else {
                "&[]".into()
            }
        },
    };

    span_lint_and_sugg(
        cx,
        USELESS_VEC,
        span,
        "useless use of `vec!`",
        "you can use a slice directly",
        snippet,
        applicability,
    );
}

/// Returns the item type of the vector (i.e., the `T` in `Vec<T>`).
fn vec_type(ty: Ty<'_>) -> Ty<'_> {
    if let ty::Adt(_, substs) = ty.kind {
        substs.type_at(0)
    } else {
        panic!("The type of `vec!` is a not a struct?");
    }
}
