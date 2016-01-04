use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::middle::def::{DefStruct, DefVariant};
use rustc_front::hir::{Expr, ExprCall, ExprLit, ExprPath, ExprStruct};
use rustc_front::hir::{Stmt, StmtSemi};

use utils::in_macro;
use utils::span_lint;

/// **What it does:** This lint `Warn`s on statements which have no effect.
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
        ExprLit(..) |
        ExprPath(..) => true,
        ExprStruct(_, ref fields, ref base) => {
            fields.iter().all(|field| has_no_effect(cx, &field.expr)) &&
            match *base {
                Some(ref base) => has_no_effect(cx, base),
                None => true,
            }
        }
        ExprCall(ref callee, ref args) => {
            let def = cx.tcx.def_map.borrow().get(&callee.id).map(|d| d.full_def());
            match def {
                Some(DefStruct(..)) |
                Some(DefVariant(..)) => args.iter().all(|arg| has_no_effect(cx, arg)),
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
