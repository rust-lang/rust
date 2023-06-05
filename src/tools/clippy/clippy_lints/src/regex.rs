use std::fmt::Display;

use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::source::snippet_opt;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::{LitKind, StrStyle};
use rustc_hir::{BorrowKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::{BytePos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks [regex](https://crates.io/crates/regex) creation
    /// (with `Regex::new`, `RegexBuilder::new`, or `RegexSet::new`) for correct
    /// regex syntax.
    ///
    /// ### Why is this bad?
    /// This will lead to a runtime panic.
    ///
    /// ### Example
    /// ```ignore
    /// Regex::new("(")
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INVALID_REGEX,
    correctness,
    "invalid regular expressions"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for trivial [regex](https://crates.io/crates/regex)
    /// creation (with `Regex::new`, `RegexBuilder::new`, or `RegexSet::new`).
    ///
    /// ### Why is this bad?
    /// Matching the regex can likely be replaced by `==` or
    /// `str::starts_with`, `str::ends_with` or `std::contains` or other `str`
    /// methods.
    ///
    /// ### Known problems
    /// If the same regex is going to be applied to multiple
    /// inputs, the precomputations done by `Regex` construction can give
    /// significantly better performance than any of the `str`-based methods.
    ///
    /// ### Example
    /// ```ignore
    /// Regex::new("^foobar")
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRIVIAL_REGEX,
    nursery,
    "trivial regular expressions"
}

declare_lint_pass!(Regex => [INVALID_REGEX, TRIVIAL_REGEX]);

impl<'tcx> LateLintPass<'tcx> for Regex {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(fun, [arg]) = expr.kind;
            if let ExprKind::Path(ref qpath) = fun.kind;
            if let Some(def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
            then {
                if match_def_path(cx, def_id, &paths::REGEX_NEW) ||
                   match_def_path(cx, def_id, &paths::REGEX_BUILDER_NEW) {
                    check_regex(cx, arg, true);
                } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_NEW) ||
                   match_def_path(cx, def_id, &paths::REGEX_BYTES_BUILDER_NEW) {
                    check_regex(cx, arg, false);
                } else if match_def_path(cx, def_id, &paths::REGEX_SET_NEW) {
                    check_set(cx, arg, true);
                } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_SET_NEW) {
                    check_set(cx, arg, false);
                }
            }
        }
    }
}

fn lint_syntax_error(cx: &LateContext<'_>, error: &regex_syntax::Error, unescaped: &str, base: Span, offset: u8) {
    let parts: Option<(_, _, &dyn Display)> = match &error {
        regex_syntax::Error::Parse(e) => Some((e.span(), e.auxiliary_span(), e.kind())),
        regex_syntax::Error::Translate(e) => Some((e.span(), None, e.kind())),
        _ => None,
    };

    let convert_span = |regex_span: &regex_syntax::ast::Span| {
        let offset = u32::from(offset);
        let start = base.lo() + BytePos(u32::try_from(regex_span.start.offset).expect("offset too large") + offset);
        let end = base.lo() + BytePos(u32::try_from(regex_span.end.offset).expect("offset too large") + offset);

        Span::new(start, end, base.ctxt(), base.parent())
    };

    if let Some((primary, auxiliary, kind)) = parts
        && let Some(literal_snippet) = snippet_opt(cx, base)
        && let Some(inner) = literal_snippet.get(offset as usize..)
        // Only convert to native rustc spans if the parsed regex matches the
        // source snippet exactly, to ensure the span offsets are correct
        && inner.get(..unescaped.len()) == Some(unescaped)
    {
        let spans = if let Some(auxiliary) = auxiliary {
            vec![convert_span(primary), convert_span(auxiliary)]
        } else {
            vec![convert_span(primary)]
        };

        span_lint(cx, INVALID_REGEX, spans, &format!("regex syntax error: {kind}"));
    } else {
        span_lint_and_help(
            cx,
            INVALID_REGEX,
            base,
            &error.to_string(),
            None,
            "consider using a raw string literal: `r\"..\"`",
        );
    }
}

fn const_str<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> Option<String> {
    constant(cx, cx.typeck_results(), e).and_then(|c| match c {
        Constant::Str(s) => Some(s),
        _ => None,
    })
}

fn is_trivial_regex(s: &regex_syntax::hir::Hir) -> Option<&'static str> {
    use regex_syntax::hir::HirKind::{Alternation, Concat, Empty, Literal, Look};
    use regex_syntax::hir::Look as HirLook;

    let is_literal = |e: &[regex_syntax::hir::Hir]| e.iter().all(|e| matches!(*e.kind(), Literal(_)));

    match *s.kind() {
        Empty | Look(_) => Some("the regex is unlikely to be useful as it is"),
        Literal(_) => Some("consider using `str::contains`"),
        Alternation(ref exprs) => {
            if exprs.iter().all(|e| matches!(e.kind(), Empty)) {
                Some("the regex is unlikely to be useful as it is")
            } else {
                None
            }
        },
        Concat(ref exprs) => match (exprs[0].kind(), exprs[exprs.len() - 1].kind()) {
            (&Look(HirLook::Start), &Look(HirLook::End)) if exprs[1..(exprs.len() - 1)].is_empty() => {
                Some("consider using `str::is_empty`")
            },
            (&Look(HirLook::Start), &Look(HirLook::End)) if is_literal(&exprs[1..(exprs.len() - 1)]) => {
                Some("consider using `==` on `str`s")
            },
            (&Look(HirLook::Start), &Literal(_)) if is_literal(&exprs[1..]) => {
                Some("consider using `str::starts_with`")
            },
            (&Literal(_), &Look(HirLook::End)) if is_literal(&exprs[1..(exprs.len() - 1)]) => {
                Some("consider using `str::ends_with`")
            },
            _ if is_literal(exprs) => Some("consider using `str::contains`"),
            _ => None,
        },
        _ => None,
    }
}

fn check_set<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, utf8: bool) {
    if_chain! {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, expr) = expr.kind;
        if let ExprKind::Array(exprs) = expr.kind;
        then {
            for expr in exprs {
                check_regex(cx, expr, utf8);
            }
        }
    }
}

fn check_regex<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, utf8: bool) {
    let mut parser = regex_syntax::ParserBuilder::new().unicode(true).utf8(utf8).build();

    if let ExprKind::Lit(lit) = expr.kind {
        if let LitKind::Str(ref r, style) = lit.node {
            let r = r.as_str();
            let offset = if let StrStyle::Raw(n) = style { 2 + n } else { 1 };
            match parser.parse(r) {
                Ok(r) => {
                    if let Some(repl) = is_trivial_regex(&r) {
                        span_lint_and_help(cx, TRIVIAL_REGEX, expr.span, "trivial regex", None, repl);
                    }
                },
                Err(e) => lint_syntax_error(cx, &e, r, expr.span, offset),
            }
        }
    } else if let Some(r) = const_str(cx, expr) {
        match parser.parse(&r) {
            Ok(r) => {
                if let Some(repl) = is_trivial_regex(&r) {
                    span_lint_and_help(cx, TRIVIAL_REGEX, expr.span, "trivial regex", None, repl);
                }
            },
            Err(e) => span_lint(cx, INVALID_REGEX, expr.span, &e.to_string()),
        }
    }
}
