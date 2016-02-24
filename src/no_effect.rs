use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::middle::def::Def;
use rustc_front::hir::{Expr, Expr_, Stmt, StmtSemi};
use utils::{in_macro, span_lint};

/// **What it does:** This lint checks for statements which have no effect.
///
/// **Why is this bad?** Similar to dead code, these statements are actually executed. However, as they have no effect, all they do is make the code less readable.
///
/// **Known problems:** None.
///
/// **Example:** `0;`
declare_lint! {
    pub NO_EFFECT,
    Warn,
    "statements with no effect"
}

fn has_no_effect(cx: &LateContext, expr: &Expr) -> bool {
    if in_macro(cx, expr.span) {
        return false;
    }
    match expr.node {
        Expr_::ExprLit(..) |
        Expr_::ExprClosure(..) |
        Expr_::ExprRange(None, None) |
        Expr_::ExprPath(..) => true,
        Expr_::ExprIndex(ref a, ref b) |
        Expr_::ExprRange(Some(ref a), Some(ref b)) |
        Expr_::ExprBinary(_, ref a, ref b) => has_no_effect(cx, a) && has_no_effect(cx, b),
        Expr_::ExprVec(ref v) |
        Expr_::ExprTup(ref v) => v.iter().all(|val| has_no_effect(cx, val)),
        Expr_::ExprRange(Some(ref inner), None) |
        Expr_::ExprRange(None, Some(ref inner)) |
        Expr_::ExprRepeat(ref inner, _) |
        Expr_::ExprCast(ref inner, _) |
        Expr_::ExprType(ref inner, _) |
        Expr_::ExprUnary(_, ref inner) |
        Expr_::ExprField(ref inner, _) |
        Expr_::ExprTupField(ref inner, _) |
        Expr_::ExprAddrOf(_, ref inner) |
        Expr_::ExprBox(ref inner) => has_no_effect(cx, inner),
        Expr_::ExprStruct(_, ref fields, ref base) => {
            fields.iter().all(|field| has_no_effect(cx, &field.expr)) &&
            match *base {
                Some(ref base) => has_no_effect(cx, base),
                None => true,
            }
        }
        Expr_::ExprCall(ref callee, ref args) => {
            let def = cx.tcx.def_map.borrow().get(&callee.id).map(|d| d.full_def());
            match def {
                Some(Def::Struct(..)) |
                Some(Def::Variant(..)) => args.iter().all(|arg| has_no_effect(cx, arg)),
                _ => false,
            }
        }
        _ => false,
    }
}

#[derive(Copy, Clone)]
pub struct NoEffectPass;

impl LintPass for NoEffectPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NO_EFFECT)
    }
}

impl LateLintPass for NoEffectPass {
    fn check_stmt(&mut self, cx: &LateContext, stmt: &Stmt) {
        if let StmtSemi(ref expr, _) = stmt.node {
            if has_no_effect(cx, expr) {
                span_lint(cx, NO_EFFECT, stmt.span, "statement with no effect");
            }
        }
    }
}
