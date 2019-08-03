use crate::utils::{has_drop, in_macro_or_desugar, snippet_opt, span_lint, span_lint_and_sugg};
use rustc::hir::def::{DefKind, Res};
use rustc::hir::{BinOpKind, BlockCheckMode, Expr, ExprKind, Stmt, StmtKind, UnsafeSource};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use std::ops::Deref;

declare_clippy_lint! {
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
    pub NO_EFFECT,
    complexity,
    "statements with no effect"
}

declare_clippy_lint! {
    /// **What it does:** Checks for expression statements that can be reduced to a
    /// sub-expression.
    ///
    /// **Why is this bad?** Expressions by themselves often have no side-effects.
    /// Having such expressions reduces readability.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// compute_array()[0];
    /// ```
    pub UNNECESSARY_OPERATION,
    complexity,
    "outer expressions with no effect"
}

fn has_no_effect(cx: &LateContext<'_, '_>, expr: &Expr) -> bool {
    if in_macro_or_desugar(expr.span) {
        return false;
    }
    match expr.node {
        ExprKind::Lit(..) | ExprKind::Closure(.., _) => true,
        ExprKind::Path(..) => !has_drop(cx, cx.tables.expr_ty(expr)),
        ExprKind::Index(ref a, ref b) | ExprKind::Binary(_, ref a, ref b) => {
            has_no_effect(cx, a) && has_no_effect(cx, b)
        },
        ExprKind::Array(ref v) | ExprKind::Tup(ref v) => v.iter().all(|val| has_no_effect(cx, val)),
        ExprKind::Repeat(ref inner, _)
        | ExprKind::Cast(ref inner, _)
        | ExprKind::Type(ref inner, _)
        | ExprKind::Unary(_, ref inner)
        | ExprKind::Field(ref inner, _)
        | ExprKind::AddrOf(_, ref inner)
        | ExprKind::Box(ref inner) => has_no_effect(cx, inner),
        ExprKind::Struct(_, ref fields, ref base) => {
            !has_drop(cx, cx.tables.expr_ty(expr))
                && fields.iter().all(|field| has_no_effect(cx, &field.expr))
                && match *base {
                    Some(ref base) => has_no_effect(cx, base),
                    None => true,
                }
        },
        ExprKind::Call(ref callee, ref args) => {
            if let ExprKind::Path(ref qpath) = callee.node {
                let res = cx.tables.qpath_res(qpath, callee.hir_id);
                match res {
                    Res::Def(DefKind::Struct, ..) | Res::Def(DefKind::Variant, ..) | Res::Def(DefKind::Ctor(..), _) => {
                        !has_drop(cx, cx.tables.expr_ty(expr)) && args.iter().all(|arg| has_no_effect(cx, arg))
                    },
                    _ => false,
                }
            } else {
                false
            }
        },
        ExprKind::Block(ref block, _) => {
            block.stmts.is_empty()
                && if let Some(ref expr) = block.expr {
                    has_no_effect(cx, expr)
                } else {
                    false
                }
        },
        _ => false,
    }
}

declare_lint_pass!(NoEffect => [NO_EFFECT, UNNECESSARY_OPERATION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NoEffect {
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt) {
        if let StmtKind::Semi(ref expr) = stmt.node {
            if has_no_effect(cx, expr) {
                span_lint(cx, NO_EFFECT, stmt.span, "statement with no effect");
            } else if let Some(reduced) = reduce_expression(cx, expr) {
                let mut snippet = String::new();
                for e in reduced {
                    if in_macro_or_desugar(e.span) {
                        return;
                    }
                    if let Some(snip) = snippet_opt(cx, e.span) {
                        snippet.push_str(&snip);
                        snippet.push(';');
                    } else {
                        return;
                    }
                }
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_OPERATION,
                    stmt.span,
                    "statement can be reduced",
                    "replace it with",
                    snippet,
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn reduce_expression<'a>(cx: &LateContext<'_, '_>, expr: &'a Expr) -> Option<Vec<&'a Expr>> {
    if in_macro_or_desugar(expr.span) {
        return None;
    }
    match expr.node {
        ExprKind::Index(ref a, ref b) => Some(vec![&**a, &**b]),
        ExprKind::Binary(ref binop, ref a, ref b) if binop.node != BinOpKind::And && binop.node != BinOpKind::Or => {
            Some(vec![&**a, &**b])
        },
        ExprKind::Array(ref v) | ExprKind::Tup(ref v) => Some(v.iter().collect()),
        ExprKind::Repeat(ref inner, _)
        | ExprKind::Cast(ref inner, _)
        | ExprKind::Type(ref inner, _)
        | ExprKind::Unary(_, ref inner)
        | ExprKind::Field(ref inner, _)
        | ExprKind::AddrOf(_, ref inner)
        | ExprKind::Box(ref inner) => reduce_expression(cx, inner).or_else(|| Some(vec![inner])),
        ExprKind::Struct(_, ref fields, ref base) => {
            if has_drop(cx, cx.tables.expr_ty(expr)) {
                None
            } else {
                Some(fields.iter().map(|f| &f.expr).chain(base).map(Deref::deref).collect())
            }
        },
        ExprKind::Call(ref callee, ref args) => {
            if let ExprKind::Path(ref qpath) = callee.node {
                let res = cx.tables.qpath_res(qpath, callee.hir_id);
                match res {
                    Res::Def(DefKind::Struct, ..) | Res::Def(DefKind::Variant, ..) | Res::Def(DefKind::Ctor(..), _)
                        if !has_drop(cx, cx.tables.expr_ty(expr)) =>
                    {
                        Some(args.iter().collect())
                    },
                    _ => None,
                }
            } else {
                None
            }
        },
        ExprKind::Block(ref block, _) => {
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
