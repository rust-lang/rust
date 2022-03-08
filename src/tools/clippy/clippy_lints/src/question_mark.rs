use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{eq_expr_value, is_lang_ctor, path_to_local, path_to_local_id, peel_blocks, peel_blocks_with_stmt};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome, ResultOk};
use rustc_hir::{BindingAnnotation, Expr, ExprKind, PatKind};
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
    #[clippy::version = "pre 1.29.0"]
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
    /// ```ignore
    /// if result.is_err() {
    ///     return result;
    /// }
    /// ```
    ///
    /// If it matches, it will suggest to use the question mark operator instead
    fn check_is_none_or_err_and_early_return(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr);
            if let ExprKind::MethodCall(segment, args, _) = &cond.kind;
            if let Some(subject) = args.get(0);
            if (Self::option_check_and_early_return(cx, subject, then) && segment.ident.name == sym!(is_none)) ||
                (Self::result_check_and_early_return(cx, subject, then) && segment.ident.name == sym!(is_err));
            then {
                let mut applicability = Applicability::MachineApplicable;
                let receiver_str = &Sugg::hir_with_applicability(cx, subject, "..", &mut applicability);
                let mut replacement: Option<String> = None;
                if let Some(else_inner) = r#else {
                    if eq_expr_value(cx, subject, peel_blocks(else_inner)) {
                        replacement = Some(format!("Some({}?)", receiver_str));
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

    fn check_if_let_some_or_err_and_early_return(cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else: Some(if_else) })
                = higher::IfLet::hir(cx, expr);
            if let PatKind::TupleStruct(ref path1, fields, None) = let_pat.kind;
            if (Self::option_check_and_early_return(cx, let_expr, if_else) && is_lang_ctor(cx, path1, OptionSome)) ||
                (Self::result_check_and_early_return(cx, let_expr, if_else) && is_lang_ctor(cx, path1, ResultOk));

            if let PatKind::Binding(annot, bind_id, _, _) = fields[0].kind;
            let by_ref = matches!(annot, BindingAnnotation::Ref | BindingAnnotation::RefMut);
            if path_to_local_id(peel_blocks(if_then), bind_id);
            then {
                let mut applicability = Applicability::MachineApplicable;
                let receiver_str = snippet_with_applicability(cx, let_expr.span, "..", &mut applicability);
                let replacement = format!("{}{}?", receiver_str, if by_ref { ".as_ref()" } else { "" },);

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

    fn result_check_and_early_return(cx: &LateContext<'_>, expr: &Expr<'_>, nested_expr: &Expr<'_>) -> bool {
        Self::is_result(cx, expr) && Self::expression_returns_unmodified_err(cx, nested_expr, expr)
    }

    fn option_check_and_early_return(cx: &LateContext<'_>, expr: &Expr<'_>, nested_expr: &Expr<'_>) -> bool {
        Self::is_option(cx, expr) && Self::expression_returns_none(cx, nested_expr)
    }

    fn moves_by_default(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        let expr_ty = cx.typeck_results().expr_ty(expression);

        !expr_ty.is_copy_modulo_regions(cx.tcx.at(expression.span), cx.param_env)
    }

    fn is_option(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        let expr_ty = cx.typeck_results().expr_ty(expression);

        is_type_diagnostic_item(cx, expr_ty, sym::Option)
    }

    fn is_result(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        let expr_ty = cx.typeck_results().expr_ty(expression);

        is_type_diagnostic_item(cx, expr_ty, sym::Result)
    }

    fn expression_returns_none(cx: &LateContext<'_>, expression: &Expr<'_>) -> bool {
        match peel_blocks_with_stmt(expression).kind {
            ExprKind::Ret(Some(expr)) => Self::expression_returns_none(cx, expr),
            ExprKind::Path(ref qpath) => is_lang_ctor(cx, qpath, OptionNone),
            _ => false,
        }
    }

    fn expression_returns_unmodified_err(cx: &LateContext<'_>, expr: &Expr<'_>, cond_expr: &Expr<'_>) -> bool {
        match peel_blocks_with_stmt(expr).kind {
            ExprKind::Ret(Some(ret_expr)) => Self::expression_returns_unmodified_err(cx, ret_expr, cond_expr),
            ExprKind::Path(_) => path_to_local(expr).is_some() && path_to_local(expr) == path_to_local(cond_expr),
            _ => false,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for QuestionMark {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        Self::check_is_none_or_err_and_early_return(cx, expr);
        Self::check_if_let_some_or_err_and_early_return(cx, expr);
    }
}
