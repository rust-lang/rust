use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_lang_item;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::{source_map::Spanned, Span};

use super::CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    call_span: Span,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
) {
    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if cx.tcx.type_of(impl_id).is_str();
        if let ExprKind::Lit(Spanned { node: LitKind::Str(ext_literal, ..), ..}) = arg.kind;
        if (2..=6).contains(&ext_literal.as_str().len());
        let ext_str = ext_literal.as_str();
        if ext_str.starts_with('.');
        if ext_str.chars().skip(1).all(|c| c.is_uppercase() || c.is_ascii_digit())
            || ext_str.chars().skip(1).all(|c| c.is_lowercase() || c.is_ascii_digit());
        let recv_ty = cx.typeck_results().expr_ty(recv).peel_refs();
        if recv_ty.is_str() || is_type_lang_item(cx, recv_ty, LangItem::String);
        then {
            span_lint_and_help(
                cx,
                CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
                call_span,
                "case-sensitive file extension comparison",
                None,
                "consider using a case-insensitive comparison instead",
            );
        }
    }
}
