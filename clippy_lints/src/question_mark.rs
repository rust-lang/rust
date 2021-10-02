use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::is_lang_ctor;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{eq_expr_value, path_to_local_id};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{BindingAnnotation, Block, Expr, ExprKind, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions that could be replaced by the question mark operator.
    ///
    /// ### Why is this bad?
    /// Question mark usage is more idiomatic.
    ///
    /// ### Example
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
    fn check_is_none_and_early_return_none(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr);
            if let ExprKind::MethodCall(segment, _, args, _) = &cond.kind;
            if segment.ident.name == sym!(is_none);
            if Self::expression_returns_none(cx, then);
            if let Some(subject) = args.get(0);
            if Self::is_option(cx, subject);

            then {
                let mut applicability = Applicability::MachineApplicable;
                let receiver_str = &Sugg::hir_with_applicability(cx, subject, "..", &mut applicability);
                let mut replacement: Option<String> = None;
                if let Some(else_inner) = r#else {
                    if_chain! {
                        if let ExprKind::Block(block, None) = &else_inner.kind;
                        if block.stmts.is_empty();
                        if let Some(block_expr) = &block.expr;
                        if eq_expr_value(cx, subject, block_expr);
                        then {
                            replacement = Some(format!("Some({}?)", receiver_str));
                        }
                    }
                } else if Self::moves_by_default(cx, subject)
                    && !matches!(subject.kind, ExprKind::Call(..) | ExprKind::MethodCall(..))
                {
                    replacement = Some(format!("{}.as_ref()?;", receiver_str));
                } else {
                    replacement = Some(format!("{}?;", receiver_str));
                }

                if let Some(replacement_str) = replacement {
                    span_lint_and_sugg(
                        cx,
                        QUESTION_MARK,
                        expr.span,
                        "this block may be rewritten with the `?` operator",
                        "replace it with",
                        replacement_str,
                        applicability,
                    );
                }
            }
        }
    }

    fn check_if_let_some_and_early_return_none(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else: Some(if_else) })
                = higher::IfLet::hir(cx, expr);
            if Self::is_option(cx, let_expr);

            if let PatKind::TupleStruct(ref path1, fields, None) = let_pat.kind;
            if is_lang_ctor(cx, path1, OptionSome);
            if let PatKind::Binding(annot, bind_id, _, _) = fields[0].kind;
            let by_ref = matches!(annot, BindingAnnotation::Ref | BindingAnnotation::RefMut);

            if let ExprKind::Block(block, None) = if_then.kind;
            if block.stmts.is_empty();
            if let Some(trailing_expr) = &block.expr;
            if path_to_local_id(trailing_expr, bind_id);

            if Self::expression_returns_none(cx, if_else);
            then {
                let mut applicability = Applicability::MachineApplicable;
                let receiver_str = snippet_with_applicability(cx, let_expr.span, "..", &mut applicability);
                let replacement = format!(
                    "{}{}?",
                    receiver_str,
                    if by_ref { ".as_ref()" } else { "" },
                );

                span_lint_and_sugg(
                    cx,
                    QUESTION_MARK,
                    expr.span,
                    "this if-let-else may be rewritten with the `?` operator",
                    "replace it with",
                    replacement,
                    applicability,
                );
            }
        }
    }

    fn moves_by_default(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        let expr_ty = cx.typeck_results().expr_ty(expression);

        !expr_ty.is_copy_modulo_regions(cx.tcx.at(expression.span), cx.param_env)
    }

    fn is_option(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        let expr_ty = cx.typeck_results().expr_ty(expression);

        is_type_diagnostic_item(cx, expr_ty, sym::Option)
    }

    fn expression_returns_none(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        match expression.kind {
            ExprKind::Block(block, _) => {
                if let Some(return_expression) = Self::return_expression(block) {
                    return Self::expression_returns_none(cx, return_expression);
                }

                false
            },
            ExprKind::Ret(Some(expr)) => Self::expression_returns_none(cx, expr),
            ExprKind::Path(ref qpath) => is_lang_ctor(cx, qpath, OptionNone),
            _ => false,
        }
    }

    fn return_expression<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
        // Check if last expression is a return statement. Then, return the expression
        if_chain! {
            if block.stmts.len() == 1;
            if let Some(expr) = block.stmts.iter().last();
            if let StmtKind::Semi(expr) = expr.kind;
            if let ExprKind::Ret(Some(ret_expr)) = expr.kind;

            then {
                return Some(ret_expr);
            }
        }

        // Check for `return` without a semicolon.
        if_chain! {
            if block.stmts.is_empty();
            if let Some(ExprKind::Ret(Some(ret_expr))) = block.expr.as_ref().map(|e| &e.kind);
            then {
                return Some(ret_expr);
            }
        }

        None
    }
}

impl<'tcx> LateLintPass<'tcx> for QuestionMark {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        Self::check_is_none_and_early_return_none(cx, expr);
        Self::check_if_let_some_and_early_return_none(cx, expr);
    }
}
