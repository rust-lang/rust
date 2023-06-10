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
    /// It's just unnecessary.
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
    pub NEEDLESS_RAW_STRING,
    complexity,
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
    complexity,
    "suggests reducing the number of hashes around a raw string literal"
}
impl_lint_pass!(RawStrings => [NEEDLESS_RAW_STRING, NEEDLESS_RAW_STRING_HASHES]);

pub struct RawStrings {
    pub needless_raw_string_hashes_allow_one: bool,
}

impl EarlyLintPass for RawStrings {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if !in_external_macro(cx.sess(), expr.span)
            && let ExprKind::Lit(lit) = expr.kind
            && let LitKind::StrRaw(num) | LitKind::ByteStrRaw(num) | LitKind::CStrRaw(num) = lit.kind
        {
            let prefix = match lit.kind {
                LitKind::StrRaw(..) => "r",
                LitKind::ByteStrRaw(..) => "br",
                LitKind::CStrRaw(..) => "cr",
                _ => unreachable!(),
            };
            if !snippet(cx, expr.span, prefix).trim().starts_with(prefix) {
                return;
            }

            if !lit.symbol.as_str().contains(['\\', '"']) {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_RAW_STRING,
                    expr.span,
                    "unnecessary raw string literal",
                    "try",
                    format!("{}\"{}\"", prefix.replace('r', ""), lit.symbol),
                    Applicability::MachineApplicable,
                );

                return;
            }

            #[expect(clippy::cast_possible_truncation)]
            let req = lit.symbol.as_str().as_bytes()
                .split(|&b| b == b'"')
                .skip(1)
                .map(|bs| 1 + bs.iter().take_while(|&&b| b == b'#').count() as u8)
                .max()
                .unwrap_or(0);

            if req < num {
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
