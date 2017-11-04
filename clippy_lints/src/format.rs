use rustc::hir::*;
use rustc::lint::*;
use rustc::ty;
use syntax::ast::LitKind;
use utils::paths;
use utils::{is_expn_of, match_def_path, match_type, opt_def_id, resolve_node, span_lint, walk_ptrs_ty};

/// **What it does:** Checks for the use of `format!("string literal with no
/// argument")` and `format!("{}", foo)` where `foo` is a string.
///
/// **Why is this bad?** There is no point of doing that. `format!("too")` can
/// be replaced by `"foo".to_owned()` if you really need a `String`. The even
/// worse `&format!("foo")` is often encountered in the wild. `format!("{}",
/// foo)` can be replaced by `foo.clone()` if `foo: String` or `foo.to_owned()`
/// if `foo: &str`.
///
/// **Known problems:** None.
///
/// **Examples:**
/// ```rust
/// format!("foo")
/// format!("{}", foo)
/// ```
declare_lint! {
    pub USELESS_FORMAT,
    Warn,
    "useless use of `format!`"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array![USELESS_FORMAT]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let Some(span) = is_expn_of(expr.span, "format") {
            match expr.node {
                // `format!("{}", foo)` expansion
                ExprCall(ref fun, ref args) => {
                    if_chain! {
                        if let ExprPath(ref qpath) = fun.node;
                        if args.len() == 2;
                        if let Some(fun_def_id) = opt_def_id(resolve_node(cx, qpath, fun.hir_id));
                        if match_def_path(cx.tcx, fun_def_id, &paths::FMT_ARGUMENTS_NEWV1);
                        // ensure the format string is `"{..}"` with only one argument and no text
                        if check_static_str(&args[0]);
                        // ensure the format argument is `{}` ie. Display with no fancy option
                        // and that the argument is a string
                        if check_arg_is_display(cx, &args[1]);
                        then {
                            span_lint(cx, USELESS_FORMAT, span, "useless use of `format!`");
                        }
                    }
                },
                // `format!("foo")` expansion contains `match () { () => [], }`
                ExprMatch(ref matchee, _, _) => if let ExprTup(ref tup) = matchee.node {
                    if tup.is_empty() {
                        span_lint(cx, USELESS_FORMAT, span, "useless use of `format!`");
                    }
                },
                _ => (),
            }
        }
    }
}

/// Checks if the expressions matches `&[""]`
fn check_static_str(expr: &Expr) -> bool {
    if_chain! {
        if let ExprAddrOf(_, ref expr) = expr.node; // &[""]
        if let ExprArray(ref exprs) = expr.node; // [""]
        if exprs.len() == 1;
        if let ExprLit(ref lit) = exprs[0].node;
        if let LitKind::Str(ref lit, _) = lit.node;
        then {
            return lit.as_str().is_empty();
        }
    }

    false
}

/// Checks if the expressions matches
/// ```rust,ignore
/// &match (&42,) {
/// (__arg0,) => [::std::fmt::ArgumentV1::new(__arg0,
/// ::std::fmt::Display::fmt)],
/// }
/// ```
fn check_arg_is_display(cx: &LateContext, expr: &Expr) -> bool {
    if_chain! {
        if let ExprAddrOf(_, ref expr) = expr.node;
        if let ExprMatch(_, ref arms, _) = expr.node;
        if arms.len() == 1;
        if arms[0].pats.len() == 1;
        if let PatKind::Tuple(ref pat, None) = arms[0].pats[0].node;
        if pat.len() == 1;
        if let ExprArray(ref exprs) = arms[0].body.node;
        if exprs.len() == 1;
        if let ExprCall(_, ref args) = exprs[0].node;
        if args.len() == 2;
        if let ExprPath(ref qpath) = args[1].node;
        if let Some(fun_def_id) = opt_def_id(resolve_node(cx, qpath, args[1].hir_id));
        if match_def_path(cx.tcx, fun_def_id, &paths::DISPLAY_FMT_METHOD);
        then {
            let ty = walk_ptrs_ty(cx.tables.pat_ty(&pat[0]));

            return ty.sty == ty::TyStr || match_type(cx, ty, &paths::STRING);
        }
    }

    false
}
