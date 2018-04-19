use regex_syntax;
use rustc::hir::*;
use rustc::lint::*;
use std::collections::HashSet;
use syntax::ast::{LitKind, NodeId, StrStyle};
use syntax::codemap::{BytePos, Span};
use utils::{is_expn_of, match_def_path, match_type, opt_def_id, paths, span_help_and_lint, span_lint};
use consts::{constant, Constant};

/// **What it does:** Checks [regex](https://crates.io/crates/regex) creation
/// (with `Regex::new`,`RegexBuilder::new` or `RegexSet::new`) for correct
/// regex syntax.
///
/// **Why is this bad?** This will lead to a runtime panic.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// Regex::new("|")
/// ```
declare_clippy_lint! {
    pub INVALID_REGEX,
    correctness,
    "invalid regular expressions"
}

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
/// ```rust
/// Regex::new("^foobar")
/// ```
declare_clippy_lint! {
    pub TRIVIAL_REGEX,
    style,
    "trivial regular expressions"
}

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
/// ```rust
/// regex!("foo|bar")
/// ```
declare_clippy_lint! {
    pub REGEX_MACRO,
    style,
    "use of `regex!(_)` instead of `Regex::new(_)`"
}

#[derive(Clone, Default)]
pub struct Pass {
    spans: HashSet<Span>,
    last: Option<NodeId>,
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INVALID_REGEX, REGEX_MACRO, TRIVIAL_REGEX)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_crate(&mut self, _: &LateContext<'a, 'tcx>, _: &'tcx Crate) {
        self.spans.clear();
    }

    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx Block) {
        if_chain! {
            if self.last.is_none();
            if let Some(ref expr) = block.expr;
            if match_type(cx, cx.tables.expr_ty(expr), &paths::REGEX);
            if let Some(span) = is_expn_of(expr.span, "regex");
            then {
                if !self.spans.contains(&span) {
                    span_lint(cx,
                              REGEX_MACRO,
                              span,
                              "`regex!(_)` found. \
                              Please use `Regex::new(_)`, which is faster for now.");
                    self.spans.insert(span);
                }
                self.last = Some(block.id);
            }
        }
    }

    fn check_block_post(&mut self, _: &LateContext<'a, 'tcx>, block: &'tcx Block) {
        if self.last.map_or(false, |id| block.id == id) {
            self.last = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprCall(ref fun, ref args) = expr.node;
            if let ExprPath(ref qpath) = fun.node;
            if args.len() == 1;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(qpath, fun.hir_id));
            then {
                if match_def_path(cx.tcx, def_id, &paths::REGEX_NEW) ||
                   match_def_path(cx.tcx, def_id, &paths::REGEX_BUILDER_NEW) {
                    check_regex(cx, &args[0], true);
                } else if match_def_path(cx.tcx, def_id, &paths::REGEX_BYTES_NEW) ||
                   match_def_path(cx.tcx, def_id, &paths::REGEX_BYTES_BUILDER_NEW) {
                    check_regex(cx, &args[0], false);
                } else if match_def_path(cx.tcx, def_id, &paths::REGEX_SET_NEW) {
                    check_set(cx, &args[0], true);
                } else if match_def_path(cx.tcx, def_id, &paths::REGEX_BYTES_SET_NEW) {
                    check_set(cx, &args[0], false);
                }
            }
        }
    }
}

fn str_span(base: Span, c: regex_syntax::ast::Span, offset: u16) -> Span {
    let offset = u32::from(offset);
    let end = base.lo() + BytePos(c.end.offset as u32 + offset);
    let start = base.lo() + BytePos(c.start.offset as u32 + offset);
    assert!(start <= end);
    Span::new(start, end, base.ctxt())
}

fn const_str<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) -> Option<String> {
    constant(cx, e).and_then(|(c, _)| match c {
        Constant::Str(s) => Some(s),
        _ => None,
    })
}

fn is_trivial_regex(s: &regex_syntax::hir::Hir) -> Option<&'static str> {
    use regex_syntax::hir::HirKind::*;
    use regex_syntax::hir::Anchor::*;

    let is_literal = |e: &[regex_syntax::hir::Hir]| e.iter().all(|e| match *e.kind() {
        Literal(_) => true,
        _ => false,
    });

    match *s.kind() {
        Empty |
        Anchor(_) => Some("the regex is unlikely to be useful as it is"),
        Literal(_) => Some("consider using `str::contains`"),
        Alternation(ref exprs) => if exprs.iter().all(|e| e.kind().is_empty()) {
            Some("the regex is unlikely to be useful as it is")
        } else {
            None
        },
        Concat(ref exprs) => match (exprs[0].kind(), exprs[exprs.len() - 1].kind()) {
            (&Anchor(StartText), &Anchor(EndText)) if exprs[1..(exprs.len() - 1)].is_empty() => Some("consider using `str::is_empty`"),
            (&Anchor(StartText), &Anchor(EndText)) if is_literal(&exprs[1..(exprs.len() - 1)]) => Some("consider using `==` on `str`s"),
            (&Anchor(StartText), &Literal(_)) if is_literal(&exprs[1..]) => Some("consider using `str::starts_with`"),
            (&Literal(_), &Anchor(EndText)) if is_literal(&exprs[1..(exprs.len() - 1)]) => Some("consider using `str::ends_with`"),
            _ if is_literal(exprs) => Some("consider using `str::contains`"),
            _ => None,
        },
        _ => None,
    }
}

fn check_set<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, utf8: bool) {
    if_chain! {
        if let ExprAddrOf(_, ref expr) = expr.node;
        if let ExprArray(ref exprs) = expr.node;
        then {
            for expr in exprs {
                check_regex(cx, expr, utf8);
            }
        }
    }
}

fn check_regex<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, utf8: bool) {
    let mut parser = regex_syntax::ParserBuilder::new()
        .unicode(utf8)
        .allow_invalid_utf8(!utf8)
        .build();

    if let ExprLit(ref lit) = expr.node {
        if let LitKind::Str(ref r, style) = lit.node {
            let r = &r.as_str();
            let offset = if let StrStyle::Raw(n) = style { 2 + n } else { 1 };
            match parser.parse(r) {
                Ok(r) => if let Some(repl) = is_trivial_regex(&r) {
                    span_help_and_lint(
                        cx,
                        TRIVIAL_REGEX,
                        expr.span,
                        "trivial regex",
                        repl,
                    );
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
                    span_lint(
                        cx,
                        INVALID_REGEX,
                        expr.span,
                        &format!("regex syntax error: {}", e),
                    );
                },
            }
        }
    } else if let Some(r) = const_str(cx, expr) {
        match parser.parse(&r) {
            Ok(r) => if let Some(repl) = is_trivial_regex(&r) {
                span_help_and_lint(
                    cx,
                    TRIVIAL_REGEX,
                    expr.span,
                    "trivial regex",
                    repl,
                );
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
                span_lint(
                    cx,
                    INVALID_REGEX,
                    expr.span,
                    &format!("regex syntax error: {}", e),
                );
            },
        }
    }
}
