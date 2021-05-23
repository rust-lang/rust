use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{BindingMode, Mutability, Pat, PatKind};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

use super::REDUNDANT_PATTERN;

pub(super) fn check(cx: &EarlyContext<'_>, pat: &Pat) {
    if let PatKind::Ident(left, ident, Some(ref right)) = pat.kind {
        let left_binding = match left {
            BindingMode::ByRef(Mutability::Mut) => "ref mut ",
            BindingMode::ByRef(Mutability::Not) => "ref ",
            BindingMode::ByValue(..) => "",
        };

        if let PatKind::Wild = right.kind {
            span_lint_and_sugg(
                cx,
                REDUNDANT_PATTERN,
                pat.span,
                &format!(
                    "the `{} @ _` pattern can be written as just `{}`",
                    ident.name, ident.name,
                ),
                "try",
                format!("{}{}", left_binding, ident.name),
                Applicability::MachineApplicable,
            );
        }
    }
}
