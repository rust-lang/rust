use super::EXPLICIT_INTO_ITER_LOOP;
use crate::utils::{snippet_with_applicability, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, method_args: &'hir [Expr<'hir>], arg: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, method_args[0].span, "_", &mut applicability);
    span_lint_and_sugg(
        cx,
        EXPLICIT_INTO_ITER_LOOP,
        arg.span,
        "it is more concise to loop over containers instead of using explicit \
            iteration methods",
        "to write this more concisely, try",
        object.to_string(),
        applicability,
    );
}
