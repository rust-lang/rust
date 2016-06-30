use rustc::lint::*;
use rustc::ty::TypeVariants;
use rustc::hir::*;
use syntax::codemap::Span;
use utils::{higher, snippet, span_lint_and_then};

/// **What it does:** This lint warns about using `&vec![..]` when using `&[..]` would be possible.
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

impl LateLintPass for Pass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        // search for `&vec![_]` expressions where the adjusted type is `&[_]`
        if_let_chain!{[
            let TypeVariants::TyRef(_, ref ty) = cx.tcx.expr_ty_adjusted(expr).sty,
            let TypeVariants::TySlice(..) = ty.ty.sty,
            let ExprAddrOf(_, ref addressee) = expr.node,
        ], {
            check_vec_macro(cx, addressee, expr.span);
        }}

        // search for `for _ in vec![…]`
        if let Some((_, arg, _)) = higher::for_loop(expr) {
            // report the error around the `vec!` not inside `<std macros>:`
            let span = cx.sess().codemap().source_callsite(arg.span);
            check_vec_macro(cx, arg, span);
        }
    }
}

fn check_vec_macro(cx: &LateContext, vec: &Expr, span: Span) {
    if let Some(vec_args) = higher::vec_macro(cx, vec) {
        let snippet = match vec_args {
            higher::VecArgs::Repeat(elem, len) => {
                format!("&[{}; {}]", snippet(cx, elem.span, "elem"), snippet(cx, len.span, "len")).into()
            }
            higher::VecArgs::Vec(args) => {
                if let Some(last) = args.iter().last() {
                    let span = Span {
                        lo: args[0].span.lo,
                        hi: last.span.hi,
                        expn_id: args[0].span.expn_id,
                    };

                    format!("&[{}]", snippet(cx, span, "..")).into()
                } else {
                    "&[]".into()
                }
            }
        };

        span_lint_and_then(cx, USELESS_VEC, span, "useless use of `vec!`", |db| {
            db.span_suggestion(span, "you can use a slice directly", snippet);
        });
    }
}

