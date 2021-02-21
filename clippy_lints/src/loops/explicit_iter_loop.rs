use super::EXPLICIT_ITER_LOOP;
use crate::utils::{snippet_with_applicability, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;

pub(super) fn lint_iter_method(cx: &LateContext<'_>, args: &[Expr<'_>], arg: &Expr<'_>, method_name: &str) {
    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, args[0].span, "_", &mut applicability);
    let muta = if method_name == "iter_mut" { "mut " } else { "" };
    span_lint_and_sugg(
        cx,
        EXPLICIT_ITER_LOOP,
        arg.span,
        "it is more concise to loop over references to containers instead of using explicit \
         iteration methods",
        "to write this more concisely, try",
        format!("&{}{}", muta, object),
        applicability,
    )
}
