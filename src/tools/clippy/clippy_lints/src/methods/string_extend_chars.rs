use crate::utils::{is_type_diagnostic_item, method_chain_args, snippet_with_applicability, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::sym;

use super::STRING_EXTEND_CHARS;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let obj_ty = cx.typeck_results().expr_ty(&args[0]).peel_refs();
    if is_type_diagnostic_item(cx, obj_ty, sym::string_type) {
        let arg = &args[1];
        if let Some(arglists) = method_chain_args(arg, &["chars"]) {
            let target = &arglists[0][0];
            let self_ty = cx.typeck_results().expr_ty(target).peel_refs();
            let ref_str = if *self_ty.kind() == ty::Str {
                ""
            } else if is_type_diagnostic_item(cx, self_ty, sym::string_type) {
                "&"
            } else {
                return;
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                STRING_EXTEND_CHARS,
                expr.span,
                "calling `.extend(_.chars())`",
                "try this",
                format!(
                    "{}.push_str({}{})",
                    snippet_with_applicability(cx, args[0].span, "..", &mut applicability),
                    ref_str,
                    snippet_with_applicability(cx, target.span, "..", &mut applicability)
                ),
                applicability,
            );
        }
    }
}
