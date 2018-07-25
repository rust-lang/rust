use rustc::hir::*;
use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use if_chain::if_chain;
use rustc::ty::{self, Ty};
use syntax::codemap::Span;
use crate::utils::{higher, is_copy, snippet, span_lint_and_sugg};
use crate::consts::constant;

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
declare_clippy_lint! {
    pub USELESS_VEC,
    perf,
    "useless `vec!`"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(USELESS_VEC)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        // search for `&vec![_]` expressions where the adjusted type is `&[_]`
        if_chain! {
            if let ty::TyRef(_, ty, _) = cx.tables.expr_ty_adjusted(expr).sty;
            if let ty::TySlice(..) = ty.sty;
            if let ExprKind::AddrOf(_, ref addressee) = expr.node;
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
                    .outer()
                    .expn_info()
                    .map(|info| info.call_site)
                    .expect("unable to get call_site");
                check_vec_macro(cx, &vec_args, span);
            }
        }
    }
}

fn check_vec_macro<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, vec_args: &higher::VecArgs<'tcx>, span: Span) {
    let snippet = match *vec_args {
        higher::VecArgs::Repeat(elem, len) => {
            if constant(cx, cx.tables, len).is_some() {
                format!("&[{}; {}]", snippet(cx, elem.span, "elem"), snippet(cx, len.span, "len"))
            } else {
                return;
            }
        },
        higher::VecArgs::Vec(args) => if let Some(last) = args.iter().last() {
            let span = args[0].span.to(last.span);

            format!("&[{}]", snippet(cx, span, ".."))
        } else {
            "&[]".into()
        },
    };

    span_lint_and_sugg(
        cx,
        USELESS_VEC,
        span,
        "useless use of `vec!`",
        "you can use a slice directly",
        snippet,
    );
}

/// Return the item type of the vector (ie. the `T` in `Vec<T>`).
fn vec_type(ty: Ty<'_>) -> Ty<'_> {
    if let ty::TyAdt(_, substs) = ty.sty {
        substs.type_at(0)
    } else {
        panic!("The type of `vec!` is a not a struct?");
    }
}
