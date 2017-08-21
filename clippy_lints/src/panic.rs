use rustc::hir::*;
use rustc::lint::*;
use syntax::ast::LitKind;
use utils::{is_direct_expn_of, match_def_path, resolve_node, paths, span_lint};

/// **What it does:** Checks for missing parameters in `panic!`.
///
/// **Why is this bad?** Contrary to the `format!` family of macros, there are
/// two forms of `panic!`: if there are no parameters given, the first argument
/// is not a format string and used literally. So while `format!("{}")` will
/// fail to compile, `panic!("{}")` will not.
///
/// **Known problems:** Should you want to use curly brackets in `panic!`
/// without any parameter, this lint will warn.
///
/// **Example:**
/// ```rust
/// panic!("This `panic!` is probably missing a parameter there: {}");
/// ```
declare_lint! {
    pub PANIC_PARAMS,
    Warn,
    "missing parameters in `panic!` calls"
}

#[allow(missing_copy_implementations)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PANIC_PARAMS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_let_chain! {[
            let ExprBlock(ref block) = expr.node,
            let Some(ref ex) = block.expr,
            let ExprCall(ref fun, ref params) = ex.node,
            params.len() == 2,
            let ExprPath(ref qpath) = fun.node,
            match_def_path(cx.tcx, resolve_node(cx, qpath, fun.hir_id).def_id(), &paths::BEGIN_PANIC),
            let ExprLit(ref lit) = params[0].node,
            is_direct_expn_of(expr.span, "panic").is_some(),
            let LitKind::Str(ref string, _) = lit.node,
            let Some(par) = string.as_str().find('{'),
            string.as_str()[par..].contains('}'),
            params[0].span.source_callee().is_none()
        ], {
            span_lint(cx, PANIC_PARAMS, params[0].span,
                      "you probably are missing some parameter in your format string");
        }}
    }
}
