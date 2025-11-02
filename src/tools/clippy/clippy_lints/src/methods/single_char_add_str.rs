use super::SINGLE_CHAR_ADD_STR;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet_with_applicability, str_literal_to_char_literal};
use rustc_ast::BorrowKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        let mut applicability = Applicability::MachineApplicable;
        let (short_name, arg, extra) = match cx.tcx.get_diagnostic_name(fn_def_id) {
            Some(sym::string_insert_str) => (
                "insert",
                &args[1],
                Some(|applicability| {
                    format!(
                        "{}, ",
                        snippet_with_applicability(cx, args[0].span, "..", applicability)
                    )
                }),
            ),
            Some(sym::string_push_str) => ("push", &args[0], None),
            _ => return,
        };

        if let Some(extension_string) = str_literal_to_char_literal(cx, arg, &mut applicability, false) {
            let base_string_snippet =
                snippet_with_applicability(cx, receiver.span.source_callsite(), "_", &mut applicability);
            span_lint_and_sugg(
                cx,
                SINGLE_CHAR_ADD_STR,
                expr.span,
                format!("calling `{short_name}_str()` using a single-character string literal"),
                format!("consider using `{short_name}` with a character literal"),
                format!(
                    "{base_string_snippet}.{short_name}({}{extension_string})",
                    extra.map_or(String::new(), |f| f(&mut applicability))
                ),
                applicability,
            );
        } else if let ExprKind::AddrOf(BorrowKind::Ref, _, inner) = arg.kind
            && let ExprKind::MethodCall(path_segment, method_arg, [], _) = inner.kind
            && path_segment.ident.name == sym::to_string
            && (is_ref_char(cx, method_arg) || is_char(cx, method_arg))
        {
            let base_string_snippet =
                snippet_with_applicability(cx, receiver.span.source_callsite(), "_", &mut applicability);
            let extension_string = match (
                snippet_with_applicability(cx, method_arg.span.source_callsite(), "_", &mut applicability),
                is_ref_char(cx, method_arg),
            ) {
                (snippet, false) => snippet,
                (snippet, true) => format!("*{snippet}").into(),
            };
            span_lint_and_sugg(
                cx,
                SINGLE_CHAR_ADD_STR,
                expr.span,
                format!("calling `{short_name}_str()` using a single-character converted to string"),
                format!("consider using `{short_name}` without `to_string()`"),
                format!(
                    "{base_string_snippet}.{short_name}({}{extension_string})",
                    extra.map_or(String::new(), |f| f(&mut applicability))
                ),
                applicability,
            );
        }
    }
}

fn is_ref_char(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(cx.typeck_results().expr_ty(expr).kind(), ty::Ref(_, ty, _) if ty.is_char())
}

fn is_char(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    cx.typeck_results().expr_ty(expr).is_char()
}
