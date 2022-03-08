use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_wild;
use rustc_hir::{Arm, PatKind};
use rustc_lint::LateContext;

use super::WILDCARD_IN_OR_PATTERNS;

pub(crate) fn check(cx: &LateContext<'_>, arms: &[Arm<'_>]) {
    for arm in arms {
        if let PatKind::Or(fields) = arm.pat.kind {
            // look for multiple fields in this arm that contains at least one Wild pattern
            if fields.len() > 1 && fields.iter().any(is_wild) {
                span_lint_and_help(
                    cx,
                    WILDCARD_IN_OR_PATTERNS,
                    arm.pat.span,
                    "wildcard pattern covers any other pattern as it will match anyway",
                    None,
                    "consider handling `_` separately",
                );
            }
        }
    }
}
