use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::Span;

use super::{OPTION_AS_REF_CLONED, method_call};

pub(super) fn check(cx: &LateContext<'_>, cloned_recv: &Expr<'_>, cloned_ident_span: Span) {
    if let Some((method @ (sym::as_ref | sym::as_mut), as_ref_recv, [], as_ref_ident_span, _)) =
        method_call(cloned_recv)
        && cx
            .typeck_results()
            .expr_ty(as_ref_recv)
            .peel_refs()
            .is_diag_item(cx, sym::Option)
    {
        span_lint_and_sugg(
            cx,
            OPTION_AS_REF_CLONED,
            as_ref_ident_span.to(cloned_ident_span),
            format!("cloning an `Option<_>` using `.{method}().cloned()`"),
            "this can be written more concisely by cloning the `Option<_>` directly",
            "clone".into(),
            Applicability::MachineApplicable,
        );
    }
}
