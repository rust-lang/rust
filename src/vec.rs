use rustc::lint::*;
use rustc::middle::ty::TypeVariants;
use rustc_front::hir::*;
use syntax::codemap::Span;
use syntax::ptr::P;
use utils::{BOX_NEW_PATH, VEC_FROM_ELEM_PATH};
use utils::{is_expn_of, match_path, snippet, span_lint_and_then};

/// **What it does:** This lint warns about using `&vec![..]` when using `&[..]` would be possible.
/// It is `Warn` by default.
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
pub struct UselessVec;

impl LintPass for UselessVec {
    fn get_lints(&self) -> LintArray {
        lint_array!(USELESS_VEC)
    }
}

impl LateLintPass for UselessVec {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        unexpand_vec(cx, expr);

        // search for `&!vec[_]` expressions where the adjusted type is `&[_]`
        if_let_chain!{[
            let TypeVariants::TyRef(_, ref ty) = cx.tcx.expr_ty_adjusted(expr).sty,
            let TypeVariants::TySlice(..) = ty.ty.sty,
            let ExprAddrOf(_, ref addressee) = expr.node,
            let Some(vec_args) = unexpand_vec(cx, addressee)
        ], {
            let snippet = match vec_args {
                VecArgs::Repeat(elem, len) => {
                    format!("&[{}; {}]", snippet(cx, elem.span, "elem"), snippet(cx, len.span, "len")).into()
                }
                VecArgs::Vec(args) => {
                    if let Some(last) = args.iter().last() {
                        let span = Span {
                            lo: args[0].span.lo,
                            hi: last.span.hi,
                            expn_id: args[0].span.expn_id,
                        };

                        format!("&[{}]", snippet(cx, span, "..")).into()
                    }
                    else {
                        "&[]".into()
                    }
                }
            };

            span_lint_and_then(cx, USELESS_VEC, expr.span, "useless use of `vec!`", |db| {
                db.span_suggestion(expr.span, "you can use a slice directly", snippet);
            });
        }}
    }
}

/// Represent the pre-expansion arguments of a `vec!` invocation.
pub enum VecArgs<'a> {
    /// `vec![elem, len]`
    Repeat(&'a P<Expr>, &'a P<Expr>),
    /// `vec![a, b, c]`
    Vec(&'a [P<Expr>]),
}

/// Returns the arguments of the `vec!` macro if this expression was expanded from `vec!`.
pub fn unexpand_vec<'e>(cx: &LateContext, expr: &'e Expr) -> Option<VecArgs<'e>> {
    if_let_chain!{[
        let ExprCall(ref fun, ref args) = expr.node,
        let ExprPath(_, ref path) = fun.node,
        is_expn_of(cx, fun.span, "vec").is_some()
    ], {
        return if match_path(path, &VEC_FROM_ELEM_PATH) && args.len() == 2 {
            // `vec![elem; size]` case
            Some(VecArgs::Repeat(&args[0], &args[1]))
        }
        else if match_path(path, &["into_vec"]) && args.len() == 1 {
            // `vec![a, b, c]` case
            if_let_chain!{[
                let ExprCall(ref fun, ref args) = args[0].node,
                let ExprPath(_, ref path) = fun.node,
                match_path(path, &BOX_NEW_PATH) && args.len() == 1,
                let ExprVec(ref args) = args[0].node
            ], {
                return Some(VecArgs::Vec(&*args));
            }}

            None
        }
        else {
            None
        };
    }}

    None
}
