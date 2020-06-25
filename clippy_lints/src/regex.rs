use crate::consts::{constant, Constant};
use crate::utils::{is_expn_of, match_def_path, match_type, paths, span_lint, span_lint_and_help};
use if_chain::if_chain;
use rustc_ast::ast::{LitKind, StrStyle};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{Block, BorrowKind, Crate, Expr, ExprKind, HirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::{BytePos, Span};
use std::convert::TryFrom;

declare_clippy_lint! {
    /// **What it does:** Checks [regex](https://crates.io/crates/regex) creation
    /// (with `Regex::new`,`RegexBuilder::new` or `RegexSet::new`) for correct
    /// regex syntax.
    ///
    /// **Why is this bad?** This will lead to a runtime panic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// Regex::new("|")
    /// ```
    pub INVALID_REGEX,
    correctness,
    "invalid regular expressions"
}

declare_clippy_lint! {
    /// **What it does:** Checks for trivial [regex](https://crates.io/crates/regex)
    /// creation (with `Regex::new`, `RegexBuilder::new` or `RegexSet::new`).
    ///
    /// **Why is this bad?** Matching the regex can likely be replaced by `==` or
    /// `str::starts_with`, `str::ends_with` or `std::contains` or other `str`
    /// methods.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// Regex::new("^foobar")
    /// ```
    pub TRIVIAL_REGEX,
    style,
    "trivial regular expressions"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usage of `regex!(_)` which (as of now) is
    /// usually slower than `Regex::new(_)` unless called in a loop (which is a bad
    /// idea anyway).
    ///
    /// **Why is this bad?** Performance, at least for now. The macro version is
    /// likely to catch up long-term, but for now the dynamic version is faster.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// regex!("foo|bar")
    /// ```
    pub REGEX_MACRO,
    style,
    "use of `regex!(_)` instead of `Regex::new(_)`"
}

#[derive(Clone, Default)]
pub struct Regex {
    spans: FxHashSet<Span>,
    last: Option<HirId>,
}

impl_lint_pass!(Regex => [INVALID_REGEX, REGEX_MACRO, TRIVIAL_REGEX]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Regex {
    fn check_crate(&mut self, _: &LateContext<'a, 'tcx>, _: &'tcx Crate<'_>) {
        self.spans.clear();
    }

    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx Block<'_>) {
        if_chain! {
            if self.last.is_none();
            if let Some(ref expr) = block.expr;
            if match_type(cx, cx.tables().expr_ty(expr), &paths::REGEX);
            if let Some(span) = is_expn_of(expr.span, "regex");
            then {
                if !self.spans.contains(&span) {
                    span_lint(
                        cx,
                        REGEX_MACRO,
                        span,
                        "`regex!(_)` found. \
                        Please use `Regex::new(_)`, which is faster for now."
                    );
                    self.spans.insert(span);
                }
                self.last = Some(block.hir_id);
            }
        }
    }

    fn check_block_post(&mut self, _: &LateContext<'a, 'tcx>, block: &'tcx Block<'_>) {
        if self.last.map_or(false, |id| block.hir_id == id) {
            self.last = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref fun, ref args) = expr.kind;
            if let ExprKind::Path(ref qpath) = fun.kind;
            if args.len() == 1;
            if let Some(def_id) = cx.tables().qpath_res(qpath, fun.hir_id).opt_def_id();
            then {
                if match_def_path(cx, def_id, &paths::REGEX_NEW) ||
                   match_def_path(cx, def_id, &paths::REGEX_BUILDER_NEW) {
                    check_regex(cx, &args[0], true);
                } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_NEW) ||
                   match_def_path(cx, def_id, &paths::REGEX_BYTES_BUILDER_NEW) {
                    check_regex(cx, &args[0], false);
                } else if match_def_path(cx, def_id, &paths::REGEX_SET_NEW) {
                    check_set(cx, &args[0], true);
                } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_SET_NEW) {
                    check_set(cx, &args[0], false);
                }
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)] // truncation very unlikely here
#[must_use]
fn str_span(base: Span, c: regex_syntax::ast::Span, offset: u16) -> Span {
    let offset = u32::from(offset);
    let end = base.lo() + BytePos(u32::try_from(c.end.offset).expect("offset too large") + offset);
    let start = base.lo() + BytePos(u32::try_from(c.start.offset).expect("offset too large") + offset);
    assert!(start <= end);
    Span::new(start, end, base.ctxt())
}

fn const_str<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, e: &'tcx Expr<'_>) -> Option<String> {
    constant(cx, cx.tables(), e).and_then(|(c, _)| match c {
        Constant::Str(s) => Some(s),
        _ => None,
    })
}

fn is_trivial_regex(s: &regex_syntax::hir::Hir) -> Option<&'static str> {
    use regex_syntax::hir::Anchor::{EndText, StartText};
    use regex_syntax::hir::HirKind::{Alternation, Anchor, Concat, Empty, Literal};

    let is_literal = |e: &[regex_syntax::hir::Hir]| {
        e.iter().all(|e| match *e.kind() {
            Literal(_) => true,
            _ => false,
        })
    };

    match *s.kind() {
        Empty | Anchor(_) => Some("the regex is unlikely to be useful as it is"),
        Literal(_) => Some("consider using `str::contains`"),
        Alternation(ref exprs) => {
            if exprs.iter().all(|e| e.kind().is_empty()) {
                Some("the regex is unlikely to be useful as it is")
            } else {
                None
            }
        },
        Concat(ref exprs) => match (exprs[0].kind(), exprs[exprs.len() - 1].kind()) {
            (&Anchor(StartText), &Anchor(EndText)) if exprs[1..(exprs.len() - 1)].is_empty() => {
                Some("consider using `str::is_empty`")
            },
            (&Anchor(StartText), &Anchor(EndText)) if is_literal(&exprs[1..(exprs.len() - 1)]) => {
                Some("consider using `==` on `str`s")
            },
            (&Anchor(StartText), &Literal(_)) if is_literal(&exprs[1..]) => Some("consider using `str::starts_with`"),
            (&Literal(_), &Anchor(EndText)) if is_literal(&exprs[1..(exprs.len() - 1)]) => {
                Some("consider using `str::ends_with`")
            },
            _ if is_literal(exprs) => Some("consider using `str::contains`"),
            _ => None,
        },
        _ => None,
    }
}

fn check_set<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>, utf8: bool) {
    if_chain! {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref expr) = expr.kind;
        if let ExprKind::Array(exprs) = expr.kind;
        then {
            for expr in exprs {
                check_regex(cx, expr, utf8);
            }
        }
    }
}

fn check_regex<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>, utf8: bool) {
    let mut parser = regex_syntax::ParserBuilder::new()
        .unicode(utf8)
        .allow_invalid_utf8(!utf8)
        .build();

    if let ExprKind::Lit(ref lit) = expr.kind {
        if let LitKind::Str(ref r, style) = lit.node {
            let r = &r.as_str();
            let offset = if let StrStyle::Raw(n) = style { 2 + n } else { 1 };
            match parser.parse(r) {
                Ok(r) => {
                    if let Some(repl) = is_trivial_regex(&r) {
                        span_lint_and_help(cx, TRIVIAL_REGEX, expr.span, "trivial regex", None, repl);
                    }
                },
                Err(regex_syntax::Error::Parse(e)) => {
                    span_lint(
                        cx,
                        INVALID_REGEX,
                        str_span(expr.span, *e.span(), offset),
                        &format!("regex syntax error: {}", e.kind()),
                    );
                },
                Err(regex_syntax::Error::Translate(e)) => {
                    span_lint(
                        cx,
                        INVALID_REGEX,
                        str_span(expr.span, *e.span(), offset),
                        &format!("regex syntax error: {}", e.kind()),
                    );
                },
                Err(e) => {
                    span_lint(cx, INVALID_REGEX, expr.span, &format!("regex syntax error: {}", e));
                },
            }
        }
    } else if let Some(r) = const_str(cx, expr) {
        match parser.parse(&r) {
            Ok(r) => {
                if let Some(repl) = is_trivial_regex(&r) {
                    span_lint_and_help(cx, TRIVIAL_REGEX, expr.span, "trivial regex", None, repl);
                }
            },
            Err(regex_syntax::Error::Parse(e)) => {
                span_lint(
                    cx,
                    INVALID_REGEX,
                    expr.span,
                    &format!("regex syntax error on position {}: {}", e.span().start.offset, e.kind()),
                );
            },
            Err(regex_syntax::Error::Translate(e)) => {
                span_lint(
                    cx,
                    INVALID_REGEX,
                    expr.span,
                    &format!("regex syntax error on position {}: {}", e.span().start.offset, e.kind()),
                );
            },
            Err(e) => {
                span_lint(cx, INVALID_REGEX, expr.span, &format!("regex syntax error: {}", e));
            },
        }
    }
}
