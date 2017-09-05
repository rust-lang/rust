use rustc::hir::*;
use rustc::lint::*;
use rustc::ty::{self, Ty};
use rustc::ty::subst::Substs;
use rustc_const_eval::ConstContext;
use syntax::codemap::Span;
use utils::{higher, is_copy, snippet, span_lint_and_sugg};

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
declare_lint! {
    pub USELESS_VEC,
    Warn,
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
        if_let_chain!{[
            let ty::TyRef(_, ref ty) = cx.tables.expr_ty_adjusted(expr).sty,
            let ty::TySlice(..) = ty.ty.sty,
            let ExprAddrOf(_, ref addressee) = expr.node,
            let Some(vec_args) = higher::vec_macro(cx, addressee),
        ], {
            check_vec_macro(cx, &vec_args, expr.span);
        }}

        // search for `for _ in vec![…]`
        if_let_chain!{[
            let Some((_, arg, _)) = higher::for_loop(expr),
            let Some(vec_args) = higher::vec_macro(cx, arg),
            is_copy(cx, vec_type(cx.tables.expr_ty_adjusted(arg))),
        ], {
            // report the error around the `vec!` not inside `<std macros>:`
            let span = arg.span.ctxt().outer().expn_info().map(|info| info.call_site).expect("unable to get call_site");
            check_vec_macro(cx, &vec_args, span);
        }}
    }
}

fn check_vec_macro(cx: &LateContext, vec_args: &higher::VecArgs, span: Span) {
    let snippet = match *vec_args {
        higher::VecArgs::Repeat(elem, len) => {
            let parent_item = cx.tcx.hir.get_parent(len.id);
            let parent_def_id = cx.tcx.hir.local_def_id(parent_item);
            let substs = Substs::identity_for_item(cx.tcx, parent_def_id);
            if ConstContext::new(cx.tcx, cx.param_env.and(substs), cx.tables)
                .eval(len)
                .is_ok()
            {
                format!("&[{}; {}]", snippet(cx, elem.span, "elem"), snippet(cx, len.span, "len")).into()
            } else {
                return;
            }
        },
        higher::VecArgs::Vec(args) => if let Some(last) = args.iter().last() {
            let span = args[0].span.to(last.span);

            format!("&[{}]", snippet(cx, span, "..")).into()
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
fn vec_type(ty: Ty) -> Ty {
    if let ty::TyAdt(_, substs) = ty.sty {
        substs.type_at(0)
    } else {
        panic!("The type of `vec!` is a not a struct?");
    }
}
