use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::hir::def::{Def, PathResolution};
use rustc::hir::{Expr, Expr_, Stmt, StmtSemi, BlockCheckMode, UnsafeSource};
use utils::{in_macro, span_lint, snippet_opt, span_lint_and_then};
use std::ops::Deref;

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

/// **What it does:** This lint checks for expression statements that can be reduced to a sub-expression
///
/// **Why is this bad?** Expressions by themselves often have no side-effects. Having such expressions reduces redability.
///
/// **Known problems:** None.
///
/// **Example:** `compute_array()[0];`
declare_lint! {
    pub UNNECESSARY_OPERATION,
    Warn,
    "outer expressions with no effect"
}

fn has_no_effect(cx: &LateContext, expr: &Expr) -> bool {
    if in_macro(cx, expr.span) {
        return false;
    }
    match expr.node {
        Expr_::ExprLit(..) |
        Expr_::ExprClosure(..) |
        Expr_::ExprPath(..) => true,
        Expr_::ExprIndex(ref a, ref b) |
        Expr_::ExprBinary(_, ref a, ref b) => has_no_effect(cx, a) && has_no_effect(cx, b),
        Expr_::ExprVec(ref v) |
        Expr_::ExprTup(ref v) => v.iter().all(|val| has_no_effect(cx, val)),
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
        Expr_::ExprBlock(ref block) => {
            block.stmts.is_empty() &&
            if let Some(ref expr) = block.expr {
                has_no_effect(cx, expr)
            } else {
                false
            }
        }
        _ => false,
    }
}

#[derive(Copy, Clone)]
pub struct NoEffectPass;

impl LintPass for NoEffectPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NO_EFFECT, UNNECESSARY_OPERATION)
    }
}

impl LateLintPass for NoEffectPass {
    fn check_stmt(&mut self, cx: &LateContext, stmt: &Stmt) {
        if let StmtSemi(ref expr, _) = stmt.node {
            if has_no_effect(cx, expr) {
                span_lint(cx, NO_EFFECT, stmt.span, "statement with no effect");
            } else if let Some(reduced) = reduce_expression(cx, expr) {
                let mut snippet = String::new();
                for e in reduced {
                    if in_macro(cx, e.span) {
                        return;
                    }
                    if let Some(snip) = snippet_opt(cx, e.span) {
                        snippet.push_str(&snip);
                        snippet.push(';');
                    } else {
                        return;
                    }
                }
                span_lint_and_then(cx, UNNECESSARY_OPERATION, stmt.span, "statement can be reduced", |db| {
                    db.span_suggestion(stmt.span, "replace it with", snippet);
                });
            }
        }
    }
}


fn reduce_expression<'a>(cx: &LateContext, expr: &'a Expr) -> Option<Vec<&'a Expr>> {
    if in_macro(cx, expr.span) {
        return None;
    }
    match expr.node {
        Expr_::ExprIndex(ref a, ref b) |
        Expr_::ExprBinary(_, ref a, ref b) => Some(vec![&**a, &**b]),
        Expr_::ExprVec(ref v) |
        Expr_::ExprTup(ref v) => Some(v.iter().map(Deref::deref).collect()),
        Expr_::ExprRepeat(ref inner, _) |
        Expr_::ExprCast(ref inner, _) |
        Expr_::ExprType(ref inner, _) |
        Expr_::ExprUnary(_, ref inner) |
        Expr_::ExprField(ref inner, _) |
        Expr_::ExprTupField(ref inner, _) |
        Expr_::ExprAddrOf(_, ref inner) |
        Expr_::ExprBox(ref inner) => reduce_expression(cx, inner).or_else(|| Some(vec![inner])),
        Expr_::ExprStruct(_, ref fields, ref base) => {
            Some(fields.iter().map(|f| &f.expr).chain(base).map(Deref::deref).collect())
        }
        Expr_::ExprCall(ref callee, ref args) => {
            match cx.tcx.def_map.borrow().get(&callee.id).map(PathResolution::full_def) {
                Some(Def::Struct(..)) |
                Some(Def::Variant(..)) => Some(args.iter().map(Deref::deref).collect()),
                _ => None,
            }
        }
        Expr_::ExprBlock(ref block) => {
            if block.stmts.is_empty() {
                block.expr.as_ref().and_then(|e| {
                    match block.rules {
                        BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) => None,
                        BlockCheckMode::DefaultBlock => Some(vec![&**e]),
                        // in case of compiler-inserted signaling blocks
                        _ => reduce_expression(cx, e),
                    }
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
