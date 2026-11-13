use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::method_chain_args;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, Lint};
use rustc_span::Symbol;

/// Wrapper fn for `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints with `unwrap()`.
pub(super) fn check(
    cx: &LateContext<'_>,
    info: &crate::methods::BinaryExprInfo<'_>,
    chain_methods: &[Symbol],
    lint: &'static Lint,
    suggest: &str,
) -> bool {
    if let Some(args) = method_chain_args(info.chain, chain_methods)
        && let hir::ExprKind::Lit(lit) = info.other.kind
        && let ast::LitKind::Char(c) = lit.node
    {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            lint,
            info.expr.span,
            format!("you should use the `{suggest}` method"),
            "like this",
            format!(
                "{}{}.{suggest}('{}')",
                if info.eq { "" } else { "!" },
                snippet_with_applicability(cx, args[0].0.span, "..", &mut applicability),
                c.escape_default()
            ),
            applicability,
        );

        true
    } else {
        false
    }
}
