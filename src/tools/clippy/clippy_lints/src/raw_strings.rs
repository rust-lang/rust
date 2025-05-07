use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, snippet_opt};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::token::LitKind;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Pos, Span};
use std::iter::once;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for raw string literals where a string literal can be used instead.
    ///
    /// ### Why restrict this?
    /// For consistent style by using simpler string literals whenever possible.
    ///
    /// However, there are many cases where using a raw string literal is more
    /// idiomatic than a string literal, so it's opt-in.
    ///
    /// ### Example
    /// ```no_run
    /// let r = r"Hello, world!";
    /// ```
    /// Use instead:
    /// ```no_run
    /// let r = "Hello, world!";
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_RAW_STRINGS,
    restriction,
    "suggests using a string literal when a raw string literal is unnecessary"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for raw string literals with an unnecessary amount of hashes around them.
    ///
    /// ### Why is this bad?
    /// It's just unnecessary, and makes it look like there's more escaping needed than is actually
    /// necessary.
    ///
    /// ### Example
    /// ```no_run
    /// let r = r###"Hello, "world"!"###;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let r = r#"Hello, "world"!"#;
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_RAW_STRING_HASHES,
    pedantic,
    "suggests reducing the number of hashes around a raw string literal"
}
impl_lint_pass!(RawStrings => [NEEDLESS_RAW_STRINGS, NEEDLESS_RAW_STRING_HASHES]);

pub struct RawStrings {
    pub allow_one_hash_in_raw_strings: bool,
}

impl RawStrings {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            allow_one_hash_in_raw_strings: conf.allow_one_hash_in_raw_strings,
        }
    }
}

impl EarlyLintPass for RawStrings {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::FormatArgs(format_args) = &expr.kind
            && !format_args.span.in_external_macro(cx.sess().source_map())
            && format_args.span.check_source_text(cx, |src| src.starts_with('r'))
            && let Some(str) = snippet_opt(cx.sess(), format_args.span)
            && let count_hash = str.bytes().skip(1).take_while(|b| *b == b'#').count()
            && let Some(str) = str.get(count_hash + 2..str.len() - count_hash - 1)
        {
            self.check_raw_string(
                cx,
                str,
                format_args.span,
                "r",
                u8::try_from(count_hash).unwrap(),
                "string",
            );
        }

        if let ExprKind::Lit(lit) = expr.kind
            && let (prefix, max) = match lit.kind {
                LitKind::StrRaw(max) => ("r", max),
                LitKind::ByteStrRaw(max) => ("br", max),
                LitKind::CStrRaw(max) => ("cr", max),
                _ => return,
            }
            && !expr.span.in_external_macro(cx.sess().source_map())
            && expr.span.check_source_text(cx, |src| src.starts_with(prefix))
        {
            self.check_raw_string(cx, lit.symbol.as_str(), expr.span, prefix, max, lit.kind.descr());
        }
    }
}

impl RawStrings {
    fn check_raw_string(
        &mut self,
        cx: &EarlyContext<'_>,
        str: &str,
        lit_span: Span,
        prefix: &str,
        max: u8,
        descr: &str,
    ) {
        if !str.contains(['\\', '"']) {
            span_lint_and_then(
                cx,
                NEEDLESS_RAW_STRINGS,
                lit_span,
                "unnecessary raw string literal",
                |diag| {
                    let (start, end) = hash_spans(lit_span, prefix.len(), 0, max);

                    // BytePos: skip over the `b` in `br`, we checked the prefix appears in the source text
                    let r_pos = lit_span.lo() + BytePos::from_usize(prefix.len() - 1);
                    let start = start.with_lo(r_pos);

                    let mut remove = vec![(start, String::new())];
                    // avoid debug ICE from empty suggestions
                    if !end.is_empty() {
                        remove.push((end, String::new()));
                    }

                    diag.multipart_suggestion_verbose(
                        format!("use a plain {descr} literal instead"),
                        remove,
                        Applicability::MachineApplicable,
                    );
                },
            );
            if !matches!(cx.get_lint_level(NEEDLESS_RAW_STRINGS).level, rustc_lint::Allow) {
                return;
            }
        }

        let mut req = {
            let mut following_quote = false;
            let mut req = 0;
            // `once` so a raw string ending in hashes is still checked
            let num = str.as_bytes().iter().chain(once(&0)).try_fold(0u8, |acc, &b| {
                match b {
                    b'"' if !following_quote => (following_quote, req) = (true, 1),
                    b'#' => req += u8::from(following_quote),
                    _ => {
                        if following_quote {
                            following_quote = false;

                            if req == max {
                                return ControlFlow::Break(req);
                            }

                            return ControlFlow::Continue(acc.max(req));
                        }
                    },
                }

                ControlFlow::Continue(acc)
            });

            match num {
                ControlFlow::Continue(num) | ControlFlow::Break(num) => num,
            }
        };
        if self.allow_one_hash_in_raw_strings {
            req = req.max(1);
        }
        if req < max {
            span_lint_and_then(
                cx,
                NEEDLESS_RAW_STRING_HASHES,
                lit_span,
                "unnecessary hashes around raw string literal",
                |diag| {
                    let (start, end) = hash_spans(lit_span, prefix.len(), req, max);

                    let message = match max - req {
                        _ if req == 0 => format!("remove all the hashes around the {descr} literal"),
                        1 => format!("remove one hash from both sides of the {descr} literal"),
                        n => format!("remove {n} hashes from both sides of the {descr} literal"),
                    };

                    diag.multipart_suggestion(
                        message,
                        vec![(start, String::new()), (end, String::new())],
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}

/// Returns spans pointing at the unneeded hashes, e.g. for a `req` of `1` and `max` of `3`:
///
/// ```ignore
/// r###".."###
///   ^^    ^^
/// ```
fn hash_spans(literal_span: Span, prefix_len: usize, req: u8, max: u8) -> (Span, Span) {
    let literal_span = literal_span.data();

    // BytePos: we checked prefix appears literally in the source text
    let hash_start = literal_span.lo + BytePos::from_usize(prefix_len);
    let hash_end = literal_span.hi;

    // BytePos: req/max are counts of the ASCII character #
    let start = Span::new(
        hash_start + BytePos(req.into()),
        hash_start + BytePos(max.into()),
        literal_span.ctxt,
        None,
    );
    let end = Span::new(
        hash_end - BytePos(req.into()),
        hash_end - BytePos(max.into()),
        literal_span.ctxt,
        None,
    );

    (start, end)
}
