use std::fmt::Display;

use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{def_path_res_with_base, find_crates, path_def_id, paths, sym};
use rustc_ast::ast::{LitKind, StrStyle};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{BorrowKind, Expr, ExprKind, OwnerId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Span};

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
    ///
    /// Use instead:
    /// ```ignore
    /// Regex::new("\(")
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
    ///
    /// Use instead:
    /// ```ignore
    /// str::starts_with("foobar")
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRIVIAL_REGEX,
    nursery,
    "trivial regular expressions"
}

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for [regex](https://crates.io/crates/regex) compilation inside a loop with a literal.
    ///
    /// ### Why is this bad?
    ///
    /// Compiling a regex is a much more expensive operation than using one, and a compiled regex can be used multiple times.
    /// This is documented as an antipattern [on the regex documentation](https://docs.rs/regex/latest/regex/#avoid-re-compiling-regexes-especially-in-a-loop)
    ///
    /// ### Example
    /// ```rust,ignore
    /// # let haystacks = [""];
    /// # const MY_REGEX: &str = "a.b";
    /// for haystack in haystacks {
    ///     let regex = regex::Regex::new(MY_REGEX).unwrap();
    ///     if regex.is_match(haystack) {
    ///         // Perform operation
    ///     }
    /// }
    /// ```
    /// can be replaced with
    /// ```rust,ignore
    /// # let haystacks = [""];
    /// # const MY_REGEX: &str = "a.b";
    /// let regex = regex::Regex::new(MY_REGEX).unwrap();
    /// for haystack in haystacks {
    ///     if regex.is_match(haystack) {
    ///         // Perform operation
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.84.0"]
    pub REGEX_CREATION_IN_LOOPS,
    perf,
    "regular expression compilation performed in a loop"
}

#[derive(Copy, Clone)]
enum RegexKind {
    Unicode,
    UnicodeSet,
    Bytes,
    BytesSet,
}

#[derive(Default)]
pub struct Regex {
    definitions: DefIdMap<RegexKind>,
    loop_stack: Vec<(OwnerId, Span)>,
}

impl_lint_pass!(Regex => [INVALID_REGEX, TRIVIAL_REGEX, REGEX_CREATION_IN_LOOPS]);

impl<'tcx> LateLintPass<'tcx> for Regex {
    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        // We don't use `match_def_path` here because that relies on matching the exact path, which changed
        // between regex 1.8 and 1.9
        //
        // `def_path_res_with_base` will resolve through re-exports but is relatively heavy, so we only
        // perform the operation once and store the results
        let regex_crates = find_crates(cx.tcx, sym::regex);
        let mut resolve = |path: &[&str], kind: RegexKind| {
            for res in def_path_res_with_base(cx.tcx, regex_crates.clone(), &path[1..]) {
                if let Some(id) = res.opt_def_id() {
                    self.definitions.insert(id, kind);
                }
            }
        };

        resolve(&paths::REGEX_NEW, RegexKind::Unicode);
        resolve(&paths::REGEX_BUILDER_NEW, RegexKind::Unicode);
        resolve(&paths::REGEX_SET_NEW, RegexKind::UnicodeSet);
        resolve(&paths::REGEX_BYTES_NEW, RegexKind::Bytes);
        resolve(&paths::REGEX_BYTES_BUILDER_NEW, RegexKind::Bytes);
        resolve(&paths::REGEX_BYTES_SET_NEW, RegexKind::BytesSet);
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(fun, [arg]) = expr.kind
            && let Some(def_id) = path_def_id(cx, fun)
            && let Some(regex_kind) = self.definitions.get(&def_id)
        {
            if let Some(&(loop_item_id, loop_span)) = self.loop_stack.last()
                && loop_item_id == fun.hir_id.owner
                && (matches!(arg.kind, ExprKind::Lit(_)) || const_str(cx, arg).is_some())
            {
                span_lint_and_help(
                    cx,
                    REGEX_CREATION_IN_LOOPS,
                    fun.span,
                    "compiling a regex in a loop",
                    Some(loop_span),
                    "move the regex construction outside this loop",
                );
            }

            match regex_kind {
                RegexKind::Unicode => check_regex(cx, arg, true),
                RegexKind::UnicodeSet => check_set(cx, arg, true),
                RegexKind::Bytes => check_regex(cx, arg, false),
                RegexKind::BytesSet => check_set(cx, arg, false),
            }
        } else if let ExprKind::Loop(block, _, _, span) = expr.kind {
            self.loop_stack.push((block.hir_id.owner, span));
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if matches!(expr.kind, ExprKind::Loop(..)) {
            self.loop_stack.pop();
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
        && let Some(literal_snippet) = base.get_source_text(cx)
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

        span_lint(cx, INVALID_REGEX, spans, format!("regex syntax error: {kind}"));
    } else {
        span_lint_and_help(
            cx,
            INVALID_REGEX,
            base,
            error.to_string(),
            None,
            "consider using a raw string literal: `r\"..\"`",
        );
    }
}

fn const_str<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> Option<String> {
    ConstEvalCtxt::new(cx).eval(e).and_then(|c| match c {
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
    if let ExprKind::AddrOf(BorrowKind::Ref, _, expr) = expr.kind
        && let ExprKind::Array(exprs) = expr.kind
    {
        for expr in exprs {
            check_regex(cx, expr, utf8);
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
            Err(e) => span_lint(cx, INVALID_REGEX, expr.span, e.to_string()),
        }
    }
}
