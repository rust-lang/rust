use clippy_utils::diagnostics::{span_lint_hir, span_lint_hir_and_then};
use clippy_utils::is_lint_allowed;
use clippy_utils::peel_blocks;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::has_drop;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{is_range_literal, BinOpKind, BlockCheckMode, Expr, ExprKind, PatKind, Stmt, StmtKind, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::ops::Deref;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for statements which have no effect.
    ///
    /// ### Why is this bad?
    /// Unlike dead code, these statements are actually
    /// executed. However, as they have no effect, all they do is make the code less
    /// readable.
    ///
    /// ### Example
    /// ```rust
    /// 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NO_EFFECT,
    complexity,
    "statements with no effect"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for binding to underscore prefixed variable without side-effects.
    ///
    /// ### Why is this bad?
    /// Unlike dead code, these bindings are actually
    /// executed. However, as they have no effect and shouldn't be used further on, all they
    /// do is make the code less readable.
    ///
    /// ### Known problems
    /// Further usage of this variable is not checked, which can lead to false positives if it is
    /// used later in the code.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _i_serve_no_purpose = 1;
    /// ```
    #[clippy::version = "1.58.0"]
    pub NO_EFFECT_UNDERSCORE_BINDING,
    pedantic,
    "binding to `_` prefixed variable with no side-effect"
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
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_OPERATION,
    complexity,
    "outer expressions with no effect"
}

declare_lint_pass!(NoEffect => [NO_EFFECT, UNNECESSARY_OPERATION, NO_EFFECT_UNDERSCORE_BINDING]);

impl<'tcx> LateLintPass<'tcx> for NoEffect {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if check_no_effect(cx, stmt) {
            return;
        }
        check_unnecessary_operation(cx, stmt);
    }
}

fn check_no_effect(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) -> bool {
    if let StmtKind::Semi(expr) = stmt.kind {
        if has_no_effect(cx, expr) {
            span_lint_hir(cx, NO_EFFECT, expr.hir_id, stmt.span, "statement with no effect");
            return true;
        }
    } else if let StmtKind::Local(local) = stmt.kind {
        if_chain! {
            if !is_lint_allowed(cx, NO_EFFECT_UNDERSCORE_BINDING, local.hir_id);
            if let Some(init) = local.init;
            if !local.pat.span.from_expansion();
            if has_no_effect(cx, init);
            if let PatKind::Binding(_, _, ident, _) = local.pat.kind;
            if ident.name.to_ident_string().starts_with('_');
            then {
                span_lint_hir(
                    cx,
                    NO_EFFECT_UNDERSCORE_BINDING,
                    init.hir_id,
                    stmt.span,
                    "binding to `_` prefixed variable with no side-effect"
                );
                return true;
            }
        }
    }
    false
}

fn has_no_effect(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if expr.span.from_expansion() {
        return false;
    }
    match peel_blocks(expr).kind {
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
                if cx.typeck_results().type_dependent_def(expr.hir_id).is_some() {
                    // type-dependent function call like `impl FnOnce for X`
                    return false;
                }
                let def_matched = matches!(
                    cx.qpath_res(qpath, callee.hir_id),
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
        _ => false,
    }
}

fn check_unnecessary_operation(cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
    if_chain! {
        if let StmtKind::Semi(expr) = stmt.kind;
        if let Some(reduced) = reduce_expression(cx, expr);
        if !&reduced.iter().any(|e| e.span.from_expansion());
        then {
            if let ExprKind::Index(..) = &expr.kind {
                let snippet = if let (Some(arr), Some(func)) =
                    (snippet_opt(cx, reduced[0].span), snippet_opt(cx, reduced[1].span))
                {
                    format!("assert!({}.len() > {});", &arr, &func)
                } else {
                    return;
                };
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
                if cx.typeck_results().type_dependent_def(expr.hir_id).is_some() {
                    // type-dependent function call like `impl FnOnce for X`
                    return None;
                }
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
