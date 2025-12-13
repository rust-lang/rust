use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::sym;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Lit, UnOp};
use rustc_lint::LateContext;
use std::cmp::Ordering;
use std::fmt;

use super::PTR_OFFSET_BY_LITERAL;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, msrv: Msrv) {
    // `pointer::add` and `pointer::wrapping_add` are only stable since 1.26.0. These functions
    // became const-stable in 1.61.0, the same version that `pointer::offset` became const-stable.
    if !msrv.meets(cx, msrvs::POINTER_ADD_SUB_METHODS) {
        return;
    }

    let ExprKind::MethodCall(method_name, recv, [arg_expr], _) = expr.kind else {
        return;
    };

    let method = match method_name.ident.name {
        sym::offset => Method::Offset,
        sym::wrapping_offset => Method::WrappingOffset,
        _ => return,
    };

    if !cx.typeck_results().expr_ty_adjusted(recv).is_raw_ptr() {
        return;
    }

    // Check if the argument to the method call is a (negated) literal.
    let Some((literal, literal_text)) = expr_as_literal(cx, arg_expr) else {
        return;
    };

    match method.suggestion(literal) {
        None => {
            let msg = format!("use of `{method}` with zero");
            span_lint_and_then(cx, PTR_OFFSET_BY_LITERAL, expr.span, msg, |diag| {
                diag.span_suggestion(
                    expr.span.with_lo(recv.span.hi()),
                    format!("remove the call to `{method}`"),
                    String::new(),
                    Applicability::MachineApplicable,
                );
            });
        },
        Some(method_suggestion) => {
            let msg = format!("use of `{method}` with a literal");
            span_lint_and_then(cx, PTR_OFFSET_BY_LITERAL, expr.span, msg, |diag| {
                diag.multipart_suggestion(
                    format!("use `{method_suggestion}` instead"),
                    vec![
                        (method_name.ident.span, method_suggestion.to_string()),
                        (arg_expr.span, literal_text),
                    ],
                    Applicability::MachineApplicable,
                );
            });
        },
    }
}

fn get_literal_bits<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<u128> {
    match expr.kind {
        ExprKind::Lit(Lit {
            node: LitKind::Int(packed_u128, _),
            ..
        }) => Some(packed_u128.get()),
        _ => None,
    }
}

// If the given expression is a (negated) literal, return its value.
fn expr_as_literal<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<(i128, String)> {
    if let Some(literal_bits) = get_literal_bits(expr) {
        // The value must fit in a isize, so we can't have overflow here.
        return Some((literal_bits.cast_signed(), format_isize_literal(cx, expr)?));
    }

    if let ExprKind::Unary(UnOp::Neg, inner) = expr.kind
        && let Some(literal_bits) = get_literal_bits(inner)
    {
        return Some((-(literal_bits.cast_signed()), format_isize_literal(cx, inner)?));
    }

    None
}

fn format_isize_literal<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<String> {
    let text = expr.span.get_source_text(cx)?;
    let text = peel_parens_str(&text);
    Some(text.trim_end_matches("isize").trim_end_matches('_').to_string())
}

fn peel_parens_str(snippet: &str) -> &str {
    let mut s = snippet.trim();
    while let Some(next) = s.strip_prefix("(").and_then(|suf| suf.strip_suffix(")")) {
        s = next.trim();
    }
    s
}

#[derive(Copy, Clone)]
enum Method {
    Offset,
    WrappingOffset,
}

impl Method {
    fn suggestion(self, literal: i128) -> Option<&'static str> {
        match Ord::cmp(&literal, &0) {
            Ordering::Greater => match self {
                Method::Offset => Some("add"),
                Method::WrappingOffset => Some("wrapping_add"),
            },
            // `ptr.offset(0)` is equivalent to `ptr`, so no adjustment is needed
            Ordering::Equal => None,
            Ordering::Less => match self {
                Method::Offset => Some("sub"),
                Method::WrappingOffset => Some("wrapping_sub"),
            },
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offset => write!(f, "offset"),
            Self::WrappingOffset => write!(f, "wrapping_offset"),
        }
    }
}
