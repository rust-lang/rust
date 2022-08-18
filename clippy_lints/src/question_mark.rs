use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{
    eq_expr_value, get_parent_node, is_else_clause, is_lang_ctor, path_to_local, path_to_local_id, peel_blocks,
    peel_blocks_with_stmt,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome, ResultErr, ResultOk};
use rustc_hir::{BindingAnnotation, Expr, ExprKind, Node, PatKind, PathSegment, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, symbol::Symbol};

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

enum IfBlockType<'hir> {
    /// An `if x.is_xxx() { a } else { b } ` expression.
    ///
    /// Contains: caller (x), caller_type, call_sym (is_xxx), if_then (a), if_else (b)
    IfIs(
        &'hir Expr<'hir>,
        Ty<'hir>,
        Symbol,
        &'hir Expr<'hir>,
        Option<&'hir Expr<'hir>>,
    ),
    /// An `if let Xxx(a) = b { c } else { d }` expression.
    ///
    /// Contains: let_pat_qpath (Xxx), let_pat_type, let_pat_sym (a), let_expr (b), if_then (c),
    /// if_else (d)
    IfLet(
        &'hir QPath<'hir>,
        Ty<'hir>,
        Symbol,
        &'hir Expr<'hir>,
        &'hir Expr<'hir>,
        Option<&'hir Expr<'hir>>,
    ),
}

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
fn check_is_none_or_err_and_early_return<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if_chain! {
        if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr);
        if !is_else_clause(cx.tcx, expr);
        if let ExprKind::MethodCall(segment, [caller, ..], _) = &cond.kind;
        let caller_ty = cx.typeck_results().expr_ty(caller);
        let if_block = IfBlockType::IfIs(caller, caller_ty, segment.ident.name, then, r#else);
        if is_early_return(sym::Option, cx, &if_block) || is_early_return(sym::Result, cx, &if_block);
        then {
            let mut applicability = Applicability::MachineApplicable;
            let receiver_str = snippet_with_applicability(cx, caller.span, "..", &mut applicability);
            let by_ref = !caller_ty.is_copy_modulo_regions(cx.tcx.at(caller.span), cx.param_env) &&
                !matches!(caller.kind, ExprKind::Call(..) | ExprKind::MethodCall(..));
            let sugg = if let Some(else_inner) = r#else {
                if eq_expr_value(cx, caller, peel_blocks(else_inner)) {
                    format!("Some({}?)", receiver_str)
                } else {
                    return;
                }
            } else {
                format!("{}{}?;", receiver_str, if by_ref { ".as_ref()" } else { "" })
            };

            span_lint_and_sugg(
                cx,
                QUESTION_MARK,
                expr.span,
                "this block may be rewritten with the `?` operator",
                "replace it with",
                sugg,
                applicability,
            );
        }
    }
}

fn check_if_let_some_or_err_and_early_return<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if_chain! {
        if let Some(higher::IfLet { let_pat, let_expr, if_then, if_else }) = higher::IfLet::hir(cx, expr);
        if !is_else_clause(cx.tcx, expr);
        if let PatKind::TupleStruct(ref path1, [field], None) = let_pat.kind;
        if let PatKind::Binding(annot, bind_id, ident, None) = field.kind;
        let caller_ty = cx.typeck_results().expr_ty(let_expr);
        let if_block = IfBlockType::IfLet(path1, caller_ty, ident.name, let_expr, if_then, if_else);
        if (is_early_return(sym::Option, cx, &if_block) && path_to_local_id(peel_blocks(if_then), bind_id))
            || is_early_return(sym::Result, cx, &if_block);
        if if_else.map(|e| eq_expr_value(cx, let_expr, peel_blocks(e))).filter(|e| *e).is_none();
        then {
            let mut applicability = Applicability::MachineApplicable;
            let receiver_str = snippet_with_applicability(cx, let_expr.span, "..", &mut applicability);
            let by_ref = matches!(annot, BindingAnnotation::Ref | BindingAnnotation::RefMut);
            let requires_semi = matches!(get_parent_node(cx.tcx, expr.hir_id), Some(Node::Stmt(_)));
            let sugg = format!(
                "{}{}?{}",
                receiver_str,
                if by_ref { ".as_ref()" } else { "" },
                if requires_semi { ";" } else { "" }
            );
            span_lint_and_sugg(
                cx,
                QUESTION_MARK,
                expr.span,
                "this block may be rewritten with the `?` operator",
                "replace it with",
                sugg,
                applicability,
            );
        }
    }
}

fn is_early_return(smbl: Symbol, cx: &LateContext<'_>, if_block: &IfBlockType<'_>) -> bool {
    match *if_block {
        IfBlockType::IfIs(caller, caller_ty, call_sym, if_then, _) => {
            // If the block could be identified as `if x.is_none()/is_err()`,
            // we then only need to check the if_then return to see if it is none/err.
            is_type_diagnostic_item(cx, caller_ty, smbl)
                && expr_return_none_or_err(smbl, cx, if_then, caller, None)
                && match smbl {
                    sym::Option => call_sym == sym!(is_none),
                    sym::Result => call_sym == sym!(is_err),
                    _ => false,
                }
        },
        IfBlockType::IfLet(qpath, let_expr_ty, let_pat_sym, let_expr, if_then, if_else) => {
            is_type_diagnostic_item(cx, let_expr_ty, smbl)
                && match smbl {
                    sym::Option => {
                        // We only need to check `if let Some(x) = option` not `if let None = option`,
                        // because the later one will be suggested as `if option.is_none()` thus causing conflict.
                        is_lang_ctor(cx, qpath, OptionSome)
                            && if_else.is_some()
                            && expr_return_none_or_err(smbl, cx, if_else.unwrap(), let_expr, None)
                    },
                    sym::Result => {
                        (is_lang_ctor(cx, qpath, ResultOk)
                            && if_else.is_some()
                            && expr_return_none_or_err(smbl, cx, if_else.unwrap(), let_expr, Some(let_pat_sym)))
                            || is_lang_ctor(cx, qpath, ResultErr)
                                && expr_return_none_or_err(smbl, cx, if_then, let_expr, Some(let_pat_sym))
                    },
                    _ => false,
                }
        },
    }
}

fn expr_return_none_or_err(
    smbl: Symbol,
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cond_expr: &Expr<'_>,
    err_sym: Option<Symbol>,
) -> bool {
    match peel_blocks_with_stmt(expr).kind {
        ExprKind::Ret(Some(ret_expr)) => expr_return_none_or_err(smbl, cx, ret_expr, cond_expr, err_sym),
        ExprKind::Path(ref qpath) => match smbl {
            sym::Option => is_lang_ctor(cx, qpath, OptionNone),
            sym::Result => path_to_local(expr).is_some() && path_to_local(expr) == path_to_local(cond_expr),
            _ => false,
        },
        ExprKind::Call(call_expr, args_expr) => {
            if_chain! {
                if smbl == sym::Result;
                if let ExprKind::Path(QPath::Resolved(_, path)) = &call_expr.kind;
                if let Some(segment) = path.segments.first();
                if let Some(err_sym) = err_sym;
                if let Some(arg) = args_expr.first();
                if let ExprKind::Path(QPath::Resolved(_, arg_path)) = &arg.kind;
                if let Some(PathSegment { ident, .. }) = arg_path.segments.first();
                then {
                    return segment.ident.name == sym::Err && err_sym == ident.name;
                }
            }
            false
        },
        _ => false,
    }
}

impl<'tcx> LateLintPass<'tcx> for QuestionMark {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        check_is_none_or_err_and_early_return(cx, expr);
        check_if_let_some_or_err_and_early_return(cx, expr);
    }
}
