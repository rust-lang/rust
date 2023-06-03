use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::{
    ast::{Expr, ExprKind},
    token::LitKind,
};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
declare_lint_pass!(NeedlessRawStringHashes => [NEEDLESS_RAW_STRING_HASHES]);

impl EarlyLintPass for NeedlessRawStringHashes {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if_chain! {
            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Lit(lit) = expr.kind;
            if let LitKind::StrRaw(num) | LitKind::ByteStrRaw(num) | LitKind::CStrRaw(num) = lit.kind;
            then {
                let str = lit.symbol.as_str();
                let mut lowest = 0;

                for i in (0..num).rev() {
                    if str.contains(&format!("\"{}", "#".repeat(i as usize))) {
                        lowest = i + 1;
                        break;
                    }
                }

                if lowest < num {
                    let hashes = "#".repeat(lowest as usize);
                    let prefix = match lit.kind {
                        LitKind::StrRaw(..) => "r",
                        LitKind::ByteStrRaw(..) => "br",
                        LitKind::CStrRaw(..) => "cr",
                        _ => unreachable!(),
                    };

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
}
