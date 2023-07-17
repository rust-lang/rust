use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::root_macro_call;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::{higher, in_constant};
use rustc_ast::ast::RangeLimits;
use rustc_ast::LitKind::{Byte, Char};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, PatKind, RangeEnd};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::DefId;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Suggests to use dedicated built-in methods,
    /// `is_ascii_(lowercase|uppercase|digit)` for checking on corresponding ascii range
    ///
    /// ### Why is this bad?
    /// Using the built-in functions is more readable and makes it
    /// clear that it's not a specific subset of characters, but all
    /// ASCII (lowercase|uppercase|digit) characters.
    /// ### Example
    /// ```rust
    /// fn main() {
    ///     assert!(matches!('x', 'a'..='z'));
    ///     assert!(matches!(b'X', b'A'..=b'Z'));
    ///     assert!(matches!('2', '0'..='9'));
    ///     assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    ///
    ///     ('0'..='9').contains(&'0');
    ///     ('a'..='z').contains(&'a');
    ///     ('A'..='Z').contains(&'A');
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn main() {
    ///     assert!('x'.is_ascii_lowercase());
    ///     assert!(b'X'.is_ascii_uppercase());
    ///     assert!('2'.is_ascii_digit());
    ///     assert!('x'.is_ascii_alphabetic());
    ///
    ///     '0'.is_ascii_digit();
    ///     'a'.is_ascii_lowercase();
    ///     'A'.is_ascii_uppercase();
    /// }
    /// ```
    #[clippy::version = "1.67.0"]
    pub MANUAL_IS_ASCII_CHECK,
    style,
    "use dedicated method to check ascii range"
}
impl_lint_pass!(ManualIsAsciiCheck => [MANUAL_IS_ASCII_CHECK]);

pub struct ManualIsAsciiCheck {
    msrv: Msrv,
}

impl ManualIsAsciiCheck {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

#[derive(Debug, PartialEq)]
enum CharRange {
    /// 'a'..='z' | b'a'..=b'z'
    LowerChar,
    /// 'A'..='Z' | b'A'..=b'Z'
    UpperChar,
    /// AsciiLower | AsciiUpper
    FullChar,
    /// '0..=9'
    Digit,
    Otherwise,
}

impl<'tcx> LateLintPass<'tcx> for ManualIsAsciiCheck {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !self.msrv.meets(msrvs::IS_ASCII_DIGIT) {
            return;
        }

        if in_constant(cx, expr.hir_id) && !self.msrv.meets(msrvs::IS_ASCII_DIGIT_CONST) {
            return;
        }

        if let Some(macro_call) = root_macro_call(expr.span)
            && is_matches_macro(cx, macro_call.def_id) {
            if let ExprKind::Match(recv, [arm, ..], _) = expr.kind {
                let range = check_pat(&arm.pat.kind);
                check_is_ascii(cx, macro_call.span, recv, &range);
            }
        } else if let ExprKind::MethodCall(path, receiver, [arg], ..) = expr.kind
            && path.ident.name == sym!(contains)
            && let Some(higher::Range { start: Some(start), end: Some(end), limits: RangeLimits::Closed })
            = higher::Range::hir(receiver) {
            let range = check_range(start, end);
            if let ExprKind::AddrOf(BorrowKind::Ref, _, e) = arg.kind {
                check_is_ascii(cx, expr.span, e, &range);
            } else {
                check_is_ascii(cx, expr.span, arg, &range);
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn check_is_ascii(cx: &LateContext<'_>, span: Span, recv: &Expr<'_>, range: &CharRange) {
    if let Some(sugg) = match range {
        CharRange::UpperChar => Some("is_ascii_uppercase"),
        CharRange::LowerChar => Some("is_ascii_lowercase"),
        CharRange::FullChar => Some("is_ascii_alphabetic"),
        CharRange::Digit => Some("is_ascii_digit"),
        CharRange::Otherwise => None,
    } {
        let default_snip = "..";
        let mut app = Applicability::MachineApplicable;
        let recv = Sugg::hir_with_context(cx, recv, span.ctxt(), default_snip, &mut app).maybe_par();

        span_lint_and_sugg(
            cx,
            MANUAL_IS_ASCII_CHECK,
            span,
            "manual check for common ascii range",
            "try",
            format!("{recv}.{sugg}()"),
            app,
        );
    }
}

fn check_pat(pat_kind: &PatKind<'_>) -> CharRange {
    match pat_kind {
        PatKind::Or(pats) => {
            let ranges = pats.iter().map(|p| check_pat(&p.kind)).collect::<Vec<_>>();

            if ranges.len() == 2 && ranges.contains(&CharRange::UpperChar) && ranges.contains(&CharRange::LowerChar) {
                CharRange::FullChar
            } else {
                CharRange::Otherwise
            }
        },
        PatKind::Range(Some(start), Some(end), kind) if *kind == RangeEnd::Included => check_range(start, end),
        _ => CharRange::Otherwise,
    }
}

fn check_range(start: &Expr<'_>, end: &Expr<'_>) -> CharRange {
    if let ExprKind::Lit(start_lit) = &start.kind
        && let ExprKind::Lit(end_lit) = &end.kind {
        match (&start_lit.node, &end_lit.node) {
            (Char('a'), Char('z')) | (Byte(b'a'), Byte(b'z')) => CharRange::LowerChar,
            (Char('A'), Char('Z')) | (Byte(b'A'), Byte(b'Z')) => CharRange::UpperChar,
            (Char('0'), Char('9')) | (Byte(b'0'), Byte(b'9')) => CharRange::Digit,
            _ => CharRange::Otherwise,
        }
    } else {
        CharRange::Otherwise
    }
}

fn is_matches_macro(cx: &LateContext<'_>, macro_def_id: DefId) -> bool {
    if let Some(name) = cx.tcx.get_diagnostic_name(macro_def_id) {
        return sym::matches_macro == name;
    }

    false
}
