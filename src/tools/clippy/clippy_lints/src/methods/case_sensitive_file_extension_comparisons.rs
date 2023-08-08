use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_opt};
use clippy_utils::ty::is_type_lang_item;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use rustc_span::Span;

use super::CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    call_span: Span,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
) {
    if let ExprKind::MethodCall(path_segment, ..) = recv.kind {
        if matches!(
            path_segment.ident.name.as_str(),
            "to_lowercase" | "to_uppercase" | "to_ascii_lowercase" | "to_ascii_uppercase"
        ) {
            return;
        }
    }

    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if cx.tcx.type_of(impl_id).instantiate_identity().is_str();
        if let ExprKind::Lit(Spanned { node: LitKind::Str(ext_literal, ..), ..}) = arg.kind;
        if (2..=6).contains(&ext_literal.as_str().len());
        let ext_str = ext_literal.as_str();
        if ext_str.starts_with('.');
        if ext_str.chars().skip(1).all(|c| c.is_uppercase() || c.is_ascii_digit())
            || ext_str.chars().skip(1).all(|c| c.is_lowercase() || c.is_ascii_digit());
        let recv_ty = cx.typeck_results().expr_ty(recv).peel_refs();
        if recv_ty.is_str() || is_type_lang_item(cx, recv_ty, LangItem::String);
        then {
            span_lint_and_then(
                cx,
                CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
                recv.span.to(call_span),
                "case-sensitive file extension comparison",
                |diag| {
                    diag.help("consider using a case-insensitive comparison instead");
                    if let Some(mut recv_source) = snippet_opt(cx, recv.span) {

                        if !cx.typeck_results().expr_ty(recv).is_ref() {
                            recv_source = format!("&{recv_source}");
                        }

                        let suggestion_source = reindent_multiline(
                            format!(
                                "std::path::Path::new({})
                                    .extension()
                                    .map_or(false, |ext| ext.eq_ignore_ascii_case(\"{}\"))",
                                recv_source, ext_str.strip_prefix('.').unwrap()).into(),
                            true,
                            Some(indent_of(cx, call_span).unwrap_or(0) + 4)
                        );

                        diag.span_suggestion(
                            recv.span.to(call_span),
                            "use std::path::Path",
                            suggestion_source,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            );
        }
    }
}
