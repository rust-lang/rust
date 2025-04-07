use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, MsrvStack};
use clippy_utils::source::{trim_span, walk_span_to_context};
use rustc_ast::ast::{Expr, ExprKind, LitKind, Pat, PatKind, RangeEnd, RangeLimits};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for ranges which almost include the entire range of letters from 'a' to 'z'
    /// or digits from '0' to '9', but don't because they're a half open range.
    ///
    /// ### Why is this bad?
    /// This (`'a'..'z'`) is almost certainly a typo meant to include all letters.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = 'a'..'z';
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = 'a'..='z';
    /// ```
    #[clippy::version = "1.68.0"]
    pub ALMOST_COMPLETE_RANGE,
    suspicious,
    "almost complete range"
}
impl_lint_pass!(AlmostCompleteRange => [ALMOST_COMPLETE_RANGE]);

pub struct AlmostCompleteRange {
    msrv: MsrvStack,
}
impl AlmostCompleteRange {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: MsrvStack::new(conf.msrv),
        }
    }
}
impl EarlyLintPass for AlmostCompleteRange {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        if let ExprKind::Range(Some(start), Some(end), RangeLimits::HalfOpen) = &e.kind
            && is_incomplete_range(start, end)
            && !e.span.in_external_macro(cx.sess().source_map())
        {
            span_lint_and_then(
                cx,
                ALMOST_COMPLETE_RANGE,
                e.span,
                "almost complete ascii range",
                |diag| {
                    let ctxt = e.span.ctxt();
                    if let Some(start) = walk_span_to_context(start.span, ctxt)
                        && let Some(end) = walk_span_to_context(end.span, ctxt)
                        && self.msrv.meets(msrvs::RANGE_INCLUSIVE)
                    {
                        diag.span_suggestion(
                            trim_span(cx.sess().source_map(), start.between(end)),
                            "use an inclusive range",
                            "..=".to_owned(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, p: &Pat) {
        if let PatKind::Range(Some(start), Some(end), kind) = &p.kind
            && matches!(kind.node, RangeEnd::Excluded)
            && is_incomplete_range(start, end)
            && !p.span.in_external_macro(cx.sess().source_map())
        {
            span_lint_and_then(
                cx,
                ALMOST_COMPLETE_RANGE,
                p.span,
                "almost complete ascii range",
                |diag| {
                    diag.span_suggestion(
                        kind.span,
                        "use an inclusive range",
                        if self.msrv.meets(msrvs::RANGE_INCLUSIVE) {
                            "..=".to_owned()
                        } else {
                            "...".to_owned()
                        },
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }

    extract_msrv_attr!();
}

fn is_incomplete_range(start: &Expr, end: &Expr) -> bool {
    match (&start.peel_parens().kind, &end.peel_parens().kind) {
        (&ExprKind::Lit(start_lit), &ExprKind::Lit(end_lit)) => {
            matches!(
                (LitKind::from_token_lit(start_lit), LitKind::from_token_lit(end_lit),),
                (
                    Ok(LitKind::Byte(b'a') | LitKind::Char('a')),
                    Ok(LitKind::Byte(b'z') | LitKind::Char('z'))
                ) | (
                    Ok(LitKind::Byte(b'A') | LitKind::Char('A')),
                    Ok(LitKind::Byte(b'Z') | LitKind::Char('Z')),
                ) | (
                    Ok(LitKind::Byte(b'0') | LitKind::Char('0')),
                    Ok(LitKind::Byte(b'9') | LitKind::Char('9')),
                )
            )
        },
        _ => false,
    }
}
