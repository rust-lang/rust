use if_chain::if_chain;
use rustc::hir::def::{DefKind, Res};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ptr::P;

use crate::utils::paths::*;
use crate::utils::sugg::Sugg;
use crate::utils::{higher, match_def_path, match_type, span_lint_and_then, SpanlessEq};

declare_clippy_lint! {
    /// **What it does:** Checks for expressions that could be replaced by the question mark operator.
    ///
    /// **Why is this bad?** Question mark usage is more idiomatic.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```ignore
    /// if option.is_none() {
    ///     return None;
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```ignore
    /// option?;
    /// ```
    pub QUESTION_MARK,
    style,
    "checks for expressions that could be replaced by the question mark operator"
}

declare_lint_pass!(QuestionMark => [QUESTION_MARK]);

impl QuestionMark {
    /// Checks if the given expression on the given context matches the following structure:
    ///
    /// ```ignore
    /// if option.is_none() {
    ///    return None;
    /// }
    /// ```
    ///
    /// If it matches, it will suggest to use the question mark operator instead
    fn check_is_none_and_early_return_none(cx: &LateContext<'_, '_>, expr: &Expr) {
        if_chain! {
            if let Some((if_expr, body, else_)) = higher::if_block(&expr);
            if let ExprKind::MethodCall(segment, _, args) = &if_expr.node;
            if segment.ident.name == sym!(is_none);
            if Self::expression_returns_none(cx, body);
            if let Some(subject) = args.get(0);
            if Self::is_option(cx, subject);

            then {
                let receiver_str = &Sugg::hir(cx, subject, "..");
                let mut replacement: Option<String> = None;
                if let Some(else_) = else_ {
                    if_chain! {
                        if let ExprKind::Block(block, None) = &else_.node;
                        if block.stmts.len() == 0;
                        if let Some(block_expr) = &block.expr;
                        if SpanlessEq::new(cx).ignore_fn().eq_expr(subject, block_expr);
                        then {
                            replacement = Some(format!("Some({}?)", receiver_str));
                        }
                    }
                } else if Self::moves_by_default(cx, subject) {
                        replacement = Some(format!("{}.as_ref()?;", receiver_str));
                } else {
                        replacement = Some(format!("{}?;", receiver_str));
                }

                if let Some(replacement_str) = replacement {
                    span_lint_and_then(
                        cx,
                        QUESTION_MARK,
                        expr.span,
                        "this block may be rewritten with the `?` operator",
                        |db| {
                            db.span_suggestion(
                                expr.span,
                                "replace_it_with",
                                replacement_str,
                                Applicability::MaybeIncorrect, // snippet
                            );
                        }
                    )
               }
            }
        }
    }

    fn moves_by_default(cx: &LateContext<'_, '_>, expression: &Expr) -> bool {
        let expr_ty = cx.tables.expr_ty(expression);

        !expr_ty.is_copy_modulo_regions(cx.tcx, cx.param_env, expression.span)
    }

    fn is_option(cx: &LateContext<'_, '_>, expression: &Expr) -> bool {
        let expr_ty = cx.tables.expr_ty(expression);

        match_type(cx, expr_ty, &OPTION)
    }

    fn expression_returns_none(cx: &LateContext<'_, '_>, expression: &Expr) -> bool {
        match expression.node {
            ExprKind::Block(ref block, _) => {
                if let Some(return_expression) = Self::return_expression(block) {
                    return Self::expression_returns_none(cx, &return_expression);
                }

                false
            },
            ExprKind::Ret(Some(ref expr)) => Self::expression_returns_none(cx, expr),
            ExprKind::Path(ref qp) => {
                if let Res::Def(DefKind::Ctor(def::CtorOf::Variant, def::CtorKind::Const), def_id) =
                    cx.tables.qpath_res(qp, expression.hir_id)
                {
                    return match_def_path(cx, def_id, &OPTION_NONE);
                }

                false
            },
            _ => false,
        }
    }

    fn return_expression(block: &Block) -> Option<&P<Expr>> {
        // Check if last expression is a return statement. Then, return the expression
        if_chain! {
            if block.stmts.len() == 1;
            if let Some(expr) = block.stmts.iter().last();
            if let StmtKind::Semi(ref expr) = expr.node;
            if let ExprKind::Ret(ref ret_expr) = expr.node;
            if let &Some(ref ret_expr) = ret_expr;

            then {
                return Some(ret_expr);
            }
        }

        // Check for `return` without a semicolon.
        if_chain! {
            if block.stmts.len() == 0;
            if let Some(ExprKind::Ret(Some(ret_expr))) = block.expr.as_ref().map(|e| &e.node);
            then {
                return Some(ret_expr);
            }
        }

        None
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for QuestionMark {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        Self::check_is_none_and_early_return_none(cx, expr);
    }
}
