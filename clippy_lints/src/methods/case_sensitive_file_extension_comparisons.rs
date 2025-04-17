use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{SpanRangeExt, indent_of, reindent_multiline};
use clippy_utils::ty::is_type_lang_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

use super::CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    call_span: Span,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
    msrv: Msrv,
) {
    if let ExprKind::MethodCall(path_segment, ..) = recv.kind
        && matches!(
            path_segment.ident.name.as_str(),
            "to_lowercase" | "to_uppercase" | "to_ascii_lowercase" | "to_ascii_uppercase"
        )
    {
        return;
    }

    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && cx.tcx.type_of(impl_id).instantiate_identity().is_str()
        && let ExprKind::Lit(Spanned {
            node: LitKind::Str(ext_literal, ..),
            ..
        }) = arg.kind
        && (2..=6).contains(&ext_literal.as_str().len())
        && let ext_str = ext_literal.as_str()
        && ext_str.starts_with('.')
        && (ext_str.chars().skip(1).all(|c| c.is_uppercase() || c.is_ascii_digit())
            || ext_str.chars().skip(1).all(|c| c.is_lowercase() || c.is_ascii_digit()))
        && !ext_str.chars().skip(1).all(|c| c.is_ascii_digit())
        && let recv_ty = cx.typeck_results().expr_ty(recv).peel_refs()
        && (recv_ty.is_str() || is_type_lang_item(cx, recv_ty, LangItem::String))
    {
        span_lint_and_then(
            cx,
            CASE_SENSITIVE_FILE_EXTENSION_COMPARISONS,
            recv.span.to(call_span),
            "case-sensitive file extension comparison",
            |diag| {
                diag.help("consider using a case-insensitive comparison instead");
                if let Some(recv_source) = recv.span.get_source_text(cx) {
                    let recv_source = if cx.typeck_results().expr_ty(recv).is_ref() {
                        recv_source.to_owned()
                    } else {
                        format!("&{recv_source}")
                    };

                    let suggestion_source = reindent_multiline(
                        &format!(
                            "std::path::Path::new({recv_source})
                                .extension()
                                .{}|ext| ext.eq_ignore_ascii_case(\"{}\"))",
                            if msrv.meets(cx, msrvs::OPTION_RESULT_IS_VARIANT_AND) {
                                "is_some_and("
                            } else {
                                "map_or(false, "
                            },
                            ext_str.strip_prefix('.').unwrap(),
                        ),
                        true,
                        Some(indent_of(cx, call_span).unwrap_or(0) + 4),
                    );

                    diag.span_suggestion(
                        recv.span.to(call_span),
                        "use std::path::Path",
                        suggestion_source,
                        Applicability::MaybeIncorrect,
                    );
                }
            },
        );
    }
}
