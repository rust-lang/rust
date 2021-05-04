use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::FloatTy;
use rustc_ast::ast::Lit;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::UNSEPARATED_LITERAL_SUFFIX;

pub(super) fn check(cx: &EarlyContext<'_>, lit: &Lit, float_ty: FloatTy, lit_snip: String) {
    let suffix = float_ty.name_str();
    let maybe_last_sep_idx = if let Some(val) = lit_snip.len().checked_sub(suffix.len() + 1) {
        val
    } else {
        return; // It's useless so shouldn't lint.
    };
    if lit_snip.as_bytes()[maybe_last_sep_idx] != b'_' {
        span_lint_and_sugg(
            cx,
            UNSEPARATED_LITERAL_SUFFIX,
            lit.span,
            "float type suffix should be separated by an underscore",
            "add an underscore",
            format!("{}_{}", &lit_snip[..=maybe_last_sep_idx], suffix),
            Applicability::MachineApplicable,
        );
    }
}
