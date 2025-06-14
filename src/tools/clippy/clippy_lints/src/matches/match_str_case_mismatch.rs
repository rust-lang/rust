use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sym;
use clippy_utils::ty::is_type_lang_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{Arm, Expr, ExprKind, LangItem, PatExpr, PatExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;
use rustc_span::symbol::Symbol;

use super::MATCH_STR_CASE_MISMATCH;

#[derive(Debug)]
enum CaseMethod {
    LowerCase,
    AsciiLowerCase,
    UpperCase,
    AsciiUppercase,
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, scrutinee: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) {
    if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty(scrutinee).kind()
        && let ty::Str = ty.kind()
    {
        let mut visitor = MatchExprVisitor { cx };
        if let ControlFlow::Break(case_method) = visitor.visit_expr(scrutinee)
            && let Some((bad_case_span, bad_case_sym)) = verify_case(&case_method, arms)
        {
            lint(cx, &case_method, bad_case_span, bad_case_sym.as_str());
        }
    }
}

struct MatchExprVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for MatchExprVisitor<'_, 'tcx> {
    type Result = ControlFlow<CaseMethod>;
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) -> Self::Result {
        if let ExprKind::MethodCall(segment, receiver, [], _) = ex.kind {
            let result = self.case_altered(segment.ident.name, receiver);
            if result.is_break() {
                return result;
            }
        }

        walk_expr(self, ex)
    }
}

impl MatchExprVisitor<'_, '_> {
    fn case_altered(&mut self, segment_ident: Symbol, receiver: &Expr<'_>) -> ControlFlow<CaseMethod> {
        if let Some(case_method) = get_case_method(segment_ident) {
            let ty = self.cx.typeck_results().expr_ty(receiver).peel_refs();

            if is_type_lang_item(self.cx, ty, LangItem::String) || ty.kind() == &ty::Str {
                return ControlFlow::Break(case_method);
            }
        }

        ControlFlow::Continue(())
    }
}

fn get_case_method(segment_ident: Symbol) -> Option<CaseMethod> {
    match segment_ident {
        sym::to_lowercase => Some(CaseMethod::LowerCase),
        sym::to_ascii_lowercase => Some(CaseMethod::AsciiLowerCase),
        sym::to_uppercase => Some(CaseMethod::UpperCase),
        sym::to_ascii_uppercase => Some(CaseMethod::AsciiUppercase),
        _ => None,
    }
}

fn verify_case<'a>(case_method: &'a CaseMethod, arms: &'a [Arm<'_>]) -> Option<(Span, Symbol)> {
    let case_check = match case_method {
        CaseMethod::LowerCase => |input: &str| -> bool { input.chars().all(|c| c.to_lowercase().next() == Some(c)) },
        CaseMethod::AsciiLowerCase => |input: &str| -> bool { !input.chars().any(|c| c.is_ascii_uppercase()) },
        CaseMethod::UpperCase => |input: &str| -> bool { input.chars().all(|c| c.to_uppercase().next() == Some(c)) },
        CaseMethod::AsciiUppercase => |input: &str| -> bool { !input.chars().any(|c| c.is_ascii_lowercase()) },
    };

    for arm in arms {
        if let PatKind::Expr(PatExpr {
            kind: PatExprKind::Lit { lit, negated: false },
            ..
        }) = arm.pat.kind
            && let LitKind::Str(symbol, _) = lit.node
            && let input = symbol.as_str()
            && !case_check(input)
        {
            return Some((lit.span, symbol));
        }
    }

    None
}

fn lint(cx: &LateContext<'_>, case_method: &CaseMethod, bad_case_span: Span, bad_case_str: &str) {
    let (method_str, suggestion) = match case_method {
        CaseMethod::LowerCase => ("to_lowercase", bad_case_str.to_lowercase()),
        CaseMethod::AsciiLowerCase => ("to_ascii_lowercase", bad_case_str.to_ascii_lowercase()),
        CaseMethod::UpperCase => ("to_uppercase", bad_case_str.to_uppercase()),
        CaseMethod::AsciiUppercase => ("to_ascii_uppercase", bad_case_str.to_ascii_uppercase()),
    };

    span_lint_and_sugg(
        cx,
        MATCH_STR_CASE_MISMATCH,
        bad_case_span,
        "this `match` arm has a differing case than its expression",
        format!("consider changing the case of this arm to respect `{method_str}`"),
        format!("\"{suggestion}\""),
        Applicability::MachineApplicable,
    );
}
