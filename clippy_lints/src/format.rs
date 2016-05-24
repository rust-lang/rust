use rustc::hir::*;
use rustc::hir::map::Node::NodeItem;
use rustc::lint::*;
use rustc::ty::TypeVariants;
use syntax::ast::LitKind;
use utils::paths;
use utils::{is_expn_of, match_path, match_type, span_lint, walk_ptrs_ty};

/// **What it does:** This lints about use of `format!("string literal with no argument")` and
/// `format!("{}", foo)` where `foo` is a string.
///
/// **Why is this bad?** There is no point of doing that. `format!("too")` can be replaced by `"foo".to_owned()` if you really need a `String`. The even worse `&format!("foo")` is often
/// encountered in the wild. `format!("{}", foo)` can be replaced by `foo.clone()` if `foo: String`
/// or `foo.to_owned()` is `foo: &str`.
///
/// **Known problems:** None.
///
/// **Examples:** `format!("foo")` and `format!("{}", foo)`
declare_lint! {
    pub USELESS_FORMAT,
    Warn,
    "useless use of `format!`"
}

#[derive(Copy, Clone, Debug)]
pub struct FormatMacLint;

impl LintPass for FormatMacLint {
    fn get_lints(&self) -> LintArray {
        lint_array![USELESS_FORMAT]
    }
}

impl LateLintPass for FormatMacLint {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let Some(span) = is_expn_of(cx, expr.span, "format") {
            match expr.node {
                // `format!("{}", foo)` expansion
                ExprCall(ref fun, ref args) => {
                    if_let_chain!{[
                        let ExprPath(_, ref path) = fun.node,
                        args.len() == 2,
                        match_path(path, &paths::FMT_ARGUMENTS_NEWV1),
                        // ensure the format string is `"{..}"` with only one argument and no text
                        check_static_str(cx, &args[0]),
                        // ensure the format argument is `{}` ie. Display with no fancy option
                        check_arg_is_display(cx, &args[1])
                    ], {
                        span_lint(cx, USELESS_FORMAT, span, "useless use of `format!`");
                    }}
                }
                // `format!("foo")` expansion contains `match () { () => [], }`
                ExprMatch(ref matchee, _, _) => {
                    if let ExprTup(ref tup) = matchee.node {
                        if tup.is_empty() {
                            span_lint(cx, USELESS_FORMAT, span, "useless use of `format!`");
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

/// Checks if the expressions matches
/// ```
/// { static __STATIC_FMTSTR: &[""] = _; __STATIC_FMTSTR }
/// ```
fn check_static_str(cx: &LateContext, expr: &Expr) -> bool {
    if_let_chain! {[
        let ExprBlock(ref block) = expr.node,
        block.stmts.len() == 1,
        let StmtDecl(ref decl, _) = block.stmts[0].node,
        let DeclItem(ref decl) = decl.node,
        let Some(NodeItem(decl)) = cx.tcx.map.find(decl.id),
        decl.name.as_str() == "__STATIC_FMTSTR",
        let ItemStatic(_, _, ref expr) = decl.node,
        let ExprAddrOf(_, ref expr) = expr.node, // &[""]
        let ExprVec(ref expr) = expr.node,
        expr.len() == 1,
        let ExprLit(ref lit) = expr[0].node,
        let LitKind::Str(ref lit, _) = lit.node,
        lit.is_empty()
    ], {
        return true;
    }}

    false
}

/// Checks if the expressions matches
/// ```
/// &match (&42,) {
///     (__arg0,) => [::std::fmt::ArgumentV1::new(__arg0, ::std::fmt::Display::fmt)],
/// })
/// ```
fn check_arg_is_display(cx: &LateContext, expr: &Expr) -> bool {
    if_let_chain! {[
        let ExprAddrOf(_, ref expr) = expr.node,
        let ExprMatch(_, ref arms, _) = expr.node,
        arms.len() == 1,
        arms[0].pats.len() == 1,
        let PatKind::Tup(ref pat) = arms[0].pats[0].node,
        pat.len() == 1,
        let ExprVec(ref exprs) = arms[0].body.node,
        exprs.len() == 1,
        let ExprCall(_, ref args) = exprs[0].node,
        args.len() == 2,
        let ExprPath(None, ref path) = args[1].node,
        match_path(path, &paths::DISPLAY_FMT_METHOD)
    ], {
        let ty = walk_ptrs_ty(cx.tcx.pat_ty(&pat[0]));

        return ty.sty == TypeVariants::TyStr || match_type(cx, ty, &paths::STRING);
    }}

    false
}
