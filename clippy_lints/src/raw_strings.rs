use std::{iter::once, ops::ControlFlow};

use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet};
use rustc_ast::{
    ast::{Expr, ExprKind},
    token::LitKind,
};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};

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
    style,
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

            if !str.contains(['\\', '"']) {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_RAW_STRINGS,
                    expr.span,
                    "unnecessary raw string literal",
                    "try",
                    format!("{}\"{}\"", prefix.replace('r', ""), lit.symbol),
                    Applicability::MachineApplicable,
                );

                return;
            }

            let req = {
                let mut following_quote = false;
                let mut req = 0;
                // `once` so a raw string ending in hashes is still checked
                let num = str.as_bytes().iter().chain(once(&0)).try_fold(0u8, |acc, &b| {
                    match b {
                        b'"' => (following_quote, req) = (true, 1),
                        // I'm a bit surprised the compiler didn't optimize this out, there's no
                        // branch but it still ends up doing an unnecessary comparison, it's:
                        // - cmp r9b,1h
                        // - sbb cl,-1h
                        // which will add 1 if it's true. With this change, it becomes:
                        // - add cl,r9b
                        // isn't that so much nicer?
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
                let hashes = "#".repeat(req as usize);

                span_lint_and_sugg(
                    cx,
                    NEEDLESS_RAW_STRING_HASHES,
                    expr.span,
                    "unnecessary hashes around raw string literal",
                    "try",
                    format!(r#"{prefix}{hashes}"{}"{hashes}"#, lit.symbol),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
