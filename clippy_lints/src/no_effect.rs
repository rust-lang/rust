use clippy_utils::diagnostics::{span_lint_hir, span_lint_hir_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::has_drop;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{is_range_literal, BinOpKind, BlockCheckMode, Expr, ExprKind, Stmt, StmtKind, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::ops::Deref;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for statements which have no effect.
    ///
    /// ### Why is this bad?
    /// Similar to dead code, these statements are actually
    /// executed. However, as they have no effect, all they do is make the code less
    /// readable.
    ///
    /// ### Example
    /// ```rust
    /// 0;
    /// ```
    pub NO_EFFECT,
    complexity,
    "statements with no effect"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expression statements that can be reduced to a
    /// sub-expression.
    ///
    /// ### Why is this bad?
    /// Expressions by themselves often have no side-effects.
    /// Having such expressions reduces readability.
    ///
    /// ### Example
    /// ```rust,ignore
    /// compute_array()[0];
    /// ```
    pub UNNECESSARY_OPERATION,
    complexity,
    "outer expressions with no effect"
}

fn has_no_effect(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if expr.span.from_expansion() {
        return false;
    }
    match expr.kind {
        ExprKind::Lit(..) | ExprKind::Closure(..) => true,
        ExprKind::Path(..) => !has_drop(cx, cx.typeck_results().expr_ty(expr)),
        ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) => has_no_effect(cx, a) && has_no_effect(cx, b),
        ExprKind::Array(v) | ExprKind::Tup(v) => v.iter().all(|val| has_no_effect(cx, val)),
        ExprKind::Repeat(inner, _)
        | ExprKind::Cast(inner, _)
        | ExprKind::Type(inner, _)
        | ExprKind::Unary(_, inner)
        | ExprKind::Field(inner, _)
        | ExprKind::AddrOf(_, _, inner)
        | ExprKind::Box(inner) => has_no_effect(cx, inner),
        ExprKind::Struct(_, fields, ref base) => {
            !has_drop(cx, cx.typeck_results().expr_ty(expr))
                && fields.iter().all(|field| has_no_effect(cx, field.expr))
                && base.as_ref().map_or(true, |base| has_no_effect(cx, base))
        },
        ExprKind::Call(callee, args) => {
            if let ExprKind::Path(ref qpath) = callee.kind {
                let res = cx.qpath_res(qpath, callee.hir_id);
                let def_matched = matches!(
                    res,
                    Res::Def(DefKind::Struct | DefKind::Variant | DefKind::Ctor(..), ..)
                );
                if def_matched || is_range_literal(expr) {
                    !has_drop(cx, cx.typeck_results().expr_ty(expr)) && args.iter().all(|arg| has_no_effect(cx, arg))
                } else {
                    false
                }
            } else {
                false
            }
        },
        ExprKind::Block(block, _) => {
            block.stmts.is_empty() && block.expr.as_ref().map_or(false, |expr| has_no_effect(cx, expr))
        },
        _ => false,
    }
}

declare_lint_pass!(NoEffect => [NO_EFFECT, UNNECESSARY_OPERATION]);

impl<'tcx> LateLintPass<'tcx> for NoEffect {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Semi(expr) = stmt.kind {
            if has_no_effect(cx, expr) {
                span_lint_hir(cx, NO_EFFECT, expr.hir_id, stmt.span, "statement with no effect");
            } else if let Some(reduced) = reduce_expression(cx, expr) {
                for e in &reduced {
                    if e.span.from_expansion() {
                        return;
                    }
                }
                if let ExprKind::Index(..) = &expr.kind {
                    let snippet;
                    if_chain! {
                        if let Some(arr) = snippet_opt(cx, reduced[0].span);
                        if let Some(func) = snippet_opt(cx, reduced[1].span);
                        then {
                            snippet = format!("assert!({}.len() > {});", &arr, &func);
                        } else {
                            return;
                        }
                    }
                    span_lint_hir_and_then(
                        cx,
                        UNNECESSARY_OPERATION,
                        expr.hir_id,
                        stmt.span,
                        "unnecessary operation",
                        |diag| {
                            diag.span_suggestion(
                                stmt.span,
                                "statement can be written as",
                                snippet,
                                Applicability::MaybeIncorrect,
                            );
                        },
                    );
                } else {
                    let mut snippet = String::new();
                    for e in reduced {
                        if let Some(snip) = snippet_opt(cx, e.span) {
                            snippet.push_str(&snip);
                            snippet.push(';');
                        } else {
                            return;
                        }
                    }
                    span_lint_hir_and_then(
                        cx,
                        UNNECESSARY_OPERATION,
                        expr.hir_id,
                        stmt.span,
                        "unnecessary operation",
                        |diag| {
                            diag.span_suggestion(
                                stmt.span,
                                "statement can be reduced to",
                                snippet,
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
            }
        }
    }
}

fn reduce_expression<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<Vec<&'a Expr<'a>>> {
    if expr.span.from_expansion() {
        return None;
    }
    match expr.kind {
        ExprKind::Index(a, b) => Some(vec![a, b]),
        ExprKind::Binary(ref binop, a, b) if binop.node != BinOpKind::And && binop.node != BinOpKind::Or => {
            Some(vec![a, b])
        },
        ExprKind::Array(v) | ExprKind::Tup(v) => Some(v.iter().collect()),
        ExprKind::Repeat(inner, _)
        | ExprKind::Cast(inner, _)
        | ExprKind::Type(inner, _)
        | ExprKind::Unary(_, inner)
        | ExprKind::Field(inner, _)
        | ExprKind::AddrOf(_, _, inner)
        | ExprKind::Box(inner) => reduce_expression(cx, inner).or_else(|| Some(vec![inner])),
        ExprKind::Struct(_, fields, ref base) => {
            if has_drop(cx, cx.typeck_results().expr_ty(expr)) {
                None
            } else {
                Some(fields.iter().map(|f| &f.expr).chain(base).map(Deref::deref).collect())
            }
        },
        ExprKind::Call(callee, args) => {
            if let ExprKind::Path(ref qpath) = callee.kind {
                let res = cx.qpath_res(qpath, callee.hir_id);
                match res {
                    Res::Def(DefKind::Struct | DefKind::Variant | DefKind::Ctor(..), ..)
                        if !has_drop(cx, cx.typeck_results().expr_ty(expr)) =>
                    {
                        Some(args.iter().collect())
                    },
                    _ => None,
                }
            } else {
                None
            }
        },
        ExprKind::Block(block, _) => {
            if block.stmts.is_empty() {
                block.expr.as_ref().and_then(|e| {
                    match block.rules {
                        BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) => None,
                        BlockCheckMode::DefaultBlock => Some(vec![&**e]),
                        // in case of compiler-inserted signaling blocks
                        BlockCheckMode::UnsafeBlock(_) => reduce_expression(cx, e),
                    }
                })
            } else {
                None
            }
        },
        _ => None,
    }
}
