use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::hir::def::Def;
use rustc::hir::{Expr, Expr_, Stmt, StmtSemi, BlockCheckMode, UnsafeSource, BiAnd, BiOr};
use utils::{in_macro, span_lint, snippet_opt, span_lint_and_sugg};
use std::ops::Deref;

/// **What it does:** Checks for statements which have no effect.
///
/// **Why is this bad?** Similar to dead code, these statements are actually
/// executed. However, as they have no effect, all they do is make the code less
/// readable.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// 0;
/// ```
declare_lint! {
    pub NO_EFFECT,
    Warn,
    "statements with no effect"
}

/// **What it does:** Checks for expression statements that can be reduced to a
/// sub-expression.
///
/// **Why is this bad?** Expressions by themselves often have no side-effects.
/// Having such expressions reduces readability.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// compute_array()[0];
/// ```
declare_lint! {
    pub UNNECESSARY_OPERATION,
    Warn,
    "outer expressions with no effect"
}

fn has_no_effect(cx: &LateContext, expr: &Expr) -> bool {
    if in_macro(expr.span) {
        return false;
    }
    match expr.node {
        Expr_::ExprLit(..) |
        Expr_::ExprClosure(..) |
        Expr_::ExprPath(..) => true,
        Expr_::ExprIndex(ref a, ref b) |
        Expr_::ExprBinary(_, ref a, ref b) => has_no_effect(cx, a) && has_no_effect(cx, b),
        Expr_::ExprArray(ref v) |
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
        },
        Expr_::ExprCall(ref callee, ref args) => {
            if let Expr_::ExprPath(ref qpath) = callee.node {
                let def = cx.tables.qpath_def(qpath, callee.id);
                match def {
                    Def::Struct(..) |
                    Def::Variant(..) |
                    Def::StructCtor(..) |
                    Def::VariantCtor(..) => args.iter().all(|arg| has_no_effect(cx, arg)),
                    _ => false,
                }
            } else {
                false
            }
        },
        Expr_::ExprBlock(ref block) => {
            block.stmts.is_empty() &&
            if let Some(ref expr) = block.expr {
                has_no_effect(cx, expr)
            } else {
                false
            }
        },
        _ => false,
    }
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NO_EFFECT, UNNECESSARY_OPERATION)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt) {
        if let StmtSemi(ref expr, _) = stmt.node {
            if has_no_effect(cx, expr) {
                span_lint(cx, NO_EFFECT, stmt.span, "statement with no effect");
            } else if let Some(reduced) = reduce_expression(cx, expr) {
                let mut snippet = String::new();
                for e in reduced {
                    if in_macro(e.span) {
                        return;
                    }
                    if let Some(snip) = snippet_opt(cx, e.span) {
                        snippet.push_str(&snip);
                        snippet.push(';');
                    } else {
                        return;
                    }
                }
                span_lint_and_sugg(cx,
                                   UNNECESSARY_OPERATION,
                                   stmt.span,
                                   "statement can be reduced",
                                   "replace it with",
                                   snippet);
            }
        }
    }
}


fn reduce_expression<'a>(cx: &LateContext, expr: &'a Expr) -> Option<Vec<&'a Expr>> {
    if in_macro(expr.span) {
        return None;
    }
    match expr.node {
        Expr_::ExprIndex(ref a, ref b) => Some(vec![&**a, &**b]),
        Expr_::ExprBinary(ref binop, ref a, ref b) if binop.node != BiAnd && binop.node != BiOr => {
            Some(vec![&**a, &**b])
        },
        Expr_::ExprArray(ref v) |
        Expr_::ExprTup(ref v) => Some(v.iter().collect()),
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
        },
        Expr_::ExprCall(ref callee, ref args) => {
            if let Expr_::ExprPath(ref qpath) = callee.node {
                let def = cx.tables.qpath_def(qpath, callee.id);
                match def {
                    Def::Struct(..) |
                    Def::Variant(..) |
                    Def::StructCtor(..) |
                    Def::VariantCtor(..) => Some(args.iter().collect()),
                    _ => None,
                }
            } else {
                None
            }
        },
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
        },
        _ => None,
    }
}
