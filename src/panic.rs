use rustc::lint::*;
use rustc_front::hir::*;
use syntax::ast::Lit_::LitStr;

use utils::{span_lint, in_external_macro, match_path, BEGIN_UNWIND};

/// **What it does:** Warn about missing parameters in `panic!`.
///
/// **Known problems:** Should you want to use curly brackets in `panic!` without any parameter,
/// this lint will warn.
///
/// **Example:**
/// ```
/// panic!("This panic! is probably missing a parameter there: {}");
/// ```
declare_lint!(pub PANIC_PARAMS, Warn, "missing parameters in `panic!`");

#[allow(missing_copy_implementations)]
pub struct PanicPass;

impl LintPass for PanicPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(PANIC_PARAMS)
    }
}

impl LateLintPass for PanicPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain! {[
            in_external_macro(cx, expr.span),
            let ExprBlock(ref block) = expr.node,
            let Some(ref ex) = block.expr,
            let ExprCall(ref fun, ref params) = ex.node,
            params.len() == 2,
            let ExprPath(None, ref path) = fun.node,
            match_path(path, &BEGIN_UNWIND),
            let ExprLit(ref lit) = params[0].node,
            let LitStr(ref string, _) = lit.node,
            string.contains('{'),
            let Some(sp) = cx.sess().codemap()
                             .with_expn_info(expr.span.expn_id,
                                             |info| info.map(|i| i.call_site))
        ], {

            span_lint(cx, PANIC_PARAMS, sp,
                      "You probably are missing some parameter in your `panic!` call");
        }}
    }
}
