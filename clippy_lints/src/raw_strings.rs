use std::iter::once;
use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_ast::token::LitKind;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Pos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for raw string literals where a string literal can be used instead.
    ///
    /// ### Why is this bad?
    /// It's just unnecessary, but there are many cases where using a raw string literal is more
    /// idiomatic than a string literal, so it's opt-in.
    ///
    /// ### Example
    /// ```rust
    /// let r = r"Hello, world!";
    /// ```
    /// Use instead:
    /// ```rust
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
    /// ```rust
    /// let r = r###"Hello, "world"!"###;
    /// ```
    /// Use instead:
    /// ```rust
    /// let r = r#"Hello, "world"!"#;
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_RAW_STRING_HASHES,
    pedantic,
    "suggests reducing the number of hashes around a raw string literal"
}
impl_lint_pass!(RawStrings => [NEEDLESS_RAW_STRINGS, NEEDLESS_RAW_STRING_HASHES]);

pub struct RawStrings {
    pub needless_raw_string_hashes_allow_one: bool,
}

impl EarlyLintPass for RawStrings {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if !in_external_macro(cx.sess(), expr.span)
            && let ExprKind::Lit(lit) = expr.kind
            && let LitKind::StrRaw(max) | LitKind::ByteStrRaw(max) | LitKind::CStrRaw(max) = lit.kind
        {
            let str = lit.symbol.as_str();
            let prefix = match lit.kind {
                LitKind::StrRaw(..) => "r",
                LitKind::ByteStrRaw(..) => "br",
                LitKind::CStrRaw(..) => "cr",
                _ => unreachable!(),
            };
            if !snippet(cx, expr.span, prefix).trim().starts_with(prefix) {
                return;
            }
            let descr = lit.kind.descr();

            if !str.contains(['\\', '"']) {
                span_lint_and_then(
                    cx,
                    NEEDLESS_RAW_STRINGS,
                    expr.span,
                    "unnecessary raw string literal",
                    |diag| {
                        let (start, end) = hash_spans(expr.span, prefix, 0, max);

                        // BytePos: skip over the `b` in `br`, we checked the prefix appears in the source text
                        let r_pos = expr.span.lo() + BytePos::from_usize(prefix.len() - 1);
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
                if !matches!(cx.get_lint_level(NEEDLESS_RAW_STRINGS), rustc_lint::Allow) {
                    return;
                }
            }

            let req = {
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

            if req < max {
                span_lint_and_then(
                    cx,
                    NEEDLESS_RAW_STRING_HASHES,
                    expr.span,
                    "unnecessary hashes around raw string literal",
                    |diag| {
                        let (start, end) = hash_spans(expr.span, prefix, req, max);

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
}

/// Returns spans pointing at the unneeded hashes, e.g. for a `req` of `1` and `max` of `3`:
///
/// ```ignore
/// r###".."###
///   ^^    ^^
/// ```
fn hash_spans(literal_span: Span, prefix: &str, req: u8, max: u8) -> (Span, Span) {
    let literal_span = literal_span.data();

    // BytePos: we checked prefix appears literally in the source text
    let hash_start = literal_span.lo + BytePos::from_usize(prefix.len());
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
