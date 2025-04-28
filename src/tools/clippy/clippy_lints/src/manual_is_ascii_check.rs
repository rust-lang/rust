use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::matching_root_macro_call;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::{higher, is_in_const_context, path_to_local, peel_ref_operators};
use rustc_ast::LitKind::{Byte, Char};
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Lit, Node, Param, PatExpr, PatExprKind, PatKind, RangeEnd};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Suggests to use dedicated built-in methods,
    /// `is_ascii_(lowercase|uppercase|digit|hexdigit)` for checking on corresponding
    /// ascii range
    ///
    /// ### Why is this bad?
    /// Using the built-in functions is more readable and makes it
    /// clear that it's not a specific subset of characters, but all
    /// ASCII (lowercase|uppercase|digit|hexdigit) characters.
    /// ### Example
    /// ```no_run
    /// fn main() {
    ///     assert!(matches!('x', 'a'..='z'));
    ///     assert!(matches!(b'X', b'A'..=b'Z'));
    ///     assert!(matches!('2', '0'..='9'));
    ///     assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    ///     assert!(matches!('C', '0'..='9' | 'a'..='f' | 'A'..='F'));
    ///
    ///     ('0'..='9').contains(&'0');
    ///     ('a'..='z').contains(&'a');
    ///     ('A'..='Z').contains(&'A');
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn main() {
    ///     assert!('x'.is_ascii_lowercase());
    ///     assert!(b'X'.is_ascii_uppercase());
    ///     assert!('2'.is_ascii_digit());
    ///     assert!('x'.is_ascii_alphabetic());
    ///     assert!('C'.is_ascii_hexdigit());
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
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

#[derive(Debug, PartialEq)]
enum CharRange {
    /// 'a'..='z' | b'a'..=b'z'
    LowerChar,
    /// 'A'..='Z' | b'A'..=b'Z'
    UpperChar,
    /// `AsciiLower` | `AsciiUpper`
    FullChar,
    /// '0..=9'
    Digit,
    /// 'a..=f'
    LowerHexLetter,
    /// 'A..=F'
    UpperHexLetter,
    /// '0..=9' | 'a..=f' | 'A..=F'
    HexDigit,
    Otherwise,
}

impl<'tcx> LateLintPass<'tcx> for ManualIsAsciiCheck {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !self.msrv.meets(cx, msrvs::IS_ASCII_DIGIT) {
            return;
        }

        if is_in_const_context(cx) && !self.msrv.meets(cx, msrvs::IS_ASCII_DIGIT_CONST) {
            return;
        }

        if let Some(macro_call) = matching_root_macro_call(cx, expr.span, sym::matches_macro) {
            if let ExprKind::Match(recv, [arm, ..], _) = expr.kind {
                let range = check_pat(&arm.pat.kind);
                check_is_ascii(cx, macro_call.span, recv, &range, None);
            }
        } else if let ExprKind::MethodCall(path, receiver, [arg], ..) = expr.kind
            && path.ident.name.as_str() == "contains"
            && let Some(higher::Range {
                start: Some(start),
                end: Some(end),
                limits: RangeLimits::Closed,
            }) = higher::Range::hir(receiver)
            && !matches!(cx.typeck_results().expr_ty(arg).peel_refs().kind(), ty::Param(_))
        {
            let arg = peel_ref_operators(cx, arg);
            let ty_sugg = get_ty_sugg(cx, arg);
            let range = check_expr_range(start, end);
            check_is_ascii(cx, expr.span, arg, &range, ty_sugg);
        }
    }
}

fn get_ty_sugg<'tcx>(cx: &LateContext<'tcx>, arg: &Expr<'_>) -> Option<(Span, Ty<'tcx>)> {
    let local_hid = path_to_local(arg)?;
    if let Node::Param(Param { ty_span, span, .. }) = cx.tcx.parent_hir_node(local_hid)
        // `ty_span` and `span` are the same for inferred type, thus a type suggestion must be given
        && ty_span == span
    {
        let arg_type = cx.typeck_results().expr_ty(arg);
        return Some((*ty_span, arg_type));
    }
    None
}

fn check_is_ascii(
    cx: &LateContext<'_>,
    span: Span,
    recv: &Expr<'_>,
    range: &CharRange,
    ty_sugg: Option<(Span, Ty<'_>)>,
) {
    let sugg = match range {
        CharRange::UpperChar => "is_ascii_uppercase",
        CharRange::LowerChar => "is_ascii_lowercase",
        CharRange::FullChar => "is_ascii_alphabetic",
        CharRange::Digit => "is_ascii_digit",
        CharRange::HexDigit => "is_ascii_hexdigit",
        CharRange::Otherwise | CharRange::LowerHexLetter | CharRange::UpperHexLetter => return,
    };
    let default_snip = "..";
    let mut app = Applicability::MachineApplicable;
    let recv = Sugg::hir_with_context(cx, recv, span.ctxt(), default_snip, &mut app).maybe_paren();
    let mut suggestion = vec![(span, format!("{recv}.{sugg}()"))];
    if let Some((ty_span, ty)) = ty_sugg {
        suggestion.push((ty_span, format!("{recv}: {ty}")));
    }

    span_lint_and_then(
        cx,
        MANUAL_IS_ASCII_CHECK,
        span,
        "manual check for common ascii range",
        |diag| {
            diag.multipart_suggestion("try", suggestion, app);
        },
    );
}

fn check_pat(pat_kind: &PatKind<'_>) -> CharRange {
    match pat_kind {
        PatKind::Or(pats) => {
            let ranges = pats.iter().map(|p| check_pat(&p.kind)).collect::<Vec<_>>();

            if ranges.len() == 2 && ranges.contains(&CharRange::UpperChar) && ranges.contains(&CharRange::LowerChar) {
                CharRange::FullChar
            } else if ranges.len() == 3
                && ranges.contains(&CharRange::Digit)
                && ranges.contains(&CharRange::LowerHexLetter)
                && ranges.contains(&CharRange::UpperHexLetter)
            {
                CharRange::HexDigit
            } else {
                CharRange::Otherwise
            }
        },
        PatKind::Range(Some(start), Some(end), kind) if *kind == RangeEnd::Included => check_range(start, end),
        _ => CharRange::Otherwise,
    }
}

fn check_expr_range(start: &Expr<'_>, end: &Expr<'_>) -> CharRange {
    if let ExprKind::Lit(start_lit) = &start.kind
        && let ExprKind::Lit(end_lit) = &end.kind
    {
        check_lit_range(start_lit, end_lit)
    } else {
        CharRange::Otherwise
    }
}

fn check_range(start: &PatExpr<'_>, end: &PatExpr<'_>) -> CharRange {
    if let PatExprKind::Lit {
        lit: start_lit,
        negated: false,
    } = &start.kind
        && let PatExprKind::Lit {
            lit: end_lit,
            negated: false,
        } = &end.kind
    {
        check_lit_range(start_lit, end_lit)
    } else {
        CharRange::Otherwise
    }
}

fn check_lit_range(start_lit: &Lit, end_lit: &Lit) -> CharRange {
    match (&start_lit.node, &end_lit.node) {
        (Char('a'), Char('z')) | (Byte(b'a'), Byte(b'z')) => CharRange::LowerChar,
        (Char('A'), Char('Z')) | (Byte(b'A'), Byte(b'Z')) => CharRange::UpperChar,
        (Char('a'), Char('f')) | (Byte(b'a'), Byte(b'f')) => CharRange::LowerHexLetter,
        (Char('A'), Char('F')) | (Byte(b'A'), Byte(b'F')) => CharRange::UpperHexLetter,
        (Char('0'), Char('9')) | (Byte(b'0'), Byte(b'9')) => CharRange::Digit,
        _ => CharRange::Otherwise,
    }
}
