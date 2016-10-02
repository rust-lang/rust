use rustc::lint::*;
use rustc::hir::*;
use utils::{paths, method_chain_args, span_help_and_lint, match_type};

/// **What it does:*** Checks for unnecessary `ok()` in if let.
///
/// **Why is this bad?** Calling `ok()` in if let is unnecessary, instead match on `Ok(x`
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rustc
/// for result in iter {
///     if let Some(bench) = try!(result).parse().ok() {
///         vec.push(bench)
///     }
/// }
/// ```
declare_lint! {
    pub IF_LET_SOME_RESULT,
    Warn,
    "usage of `ok()` in `if let Some(x)` statements is unnecessary, match on `Ok(expr)` instead"
}

#[derive(Copy, Clone)]
pub struct OkIfLetPass;

impl LintPass for OkIfLetPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(IF_LET_SOME_RESULT)
    }
}

impl LateLintPass for OkIfLetPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain! {[ //begin checking variables
            let ExprMatch(ref op, ref body, ref source) = expr.node, //test if expr is a match
            let MatchSource::IfLetDesugar { contains_else_clause: _ } = *source, //test if it is an If Let
            let ExprMethodCall(_, _, ref result_types) = op.node, //check is expr.ok() has type Result<T,E>.ok()
            let PatKind::TupleStruct(ref x, ref y, _)  = body[0].pats[0].node, //get operation
            let Some(_) = method_chain_args(op, &["ok"]) //test to see if using ok() methoduse std::marker::Sized;

        ], {
            let is_result_type = match_type(cx, cx.tcx.expr_ty(&result_types[0]), &paths::RESULT);
            let some_expr_string = print::pat_to_string(&y[0]);
            if print::path_to_string(x) == "Some" && is_result_type {
                span_help_and_lint(cx, IF_LET_SOME_RESULT, expr.span,
                "Matching on `Some` with `ok()` is redundant",
                &format!("Consider matching on `Ok({})` and removing the call to `ok` instead", some_expr_string)); 
            }
        }}
    }
}
