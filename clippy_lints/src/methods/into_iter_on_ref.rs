use crate::utils::{has_iter_method, match_trait_method, paths, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Span;
use rustc_span::symbol::Symbol;

use super::INTO_ITER_ON_REF;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, self_ref_ty: Ty<'_>, method_span: Span) {
    if !match_trait_method(cx, expr, &paths::INTO_ITERATOR) {
        return;
    }
    if let Some((kind, method_name)) = ty_has_iter_method(cx, self_ref_ty) {
        span_lint_and_sugg(
            cx,
            INTO_ITER_ON_REF,
            method_span,
            &format!(
                "this `.into_iter()` call is equivalent to `.{}()` and will not consume the `{}`",
                method_name, kind,
            ),
            "call directly",
            method_name.to_string(),
            Applicability::MachineApplicable,
        );
    }
}

fn ty_has_iter_method(cx: &LateContext<'_>, self_ref_ty: Ty<'_>) -> Option<(Symbol, &'static str)> {
    has_iter_method(cx, self_ref_ty).map(|ty_name| {
        let mutbl = match self_ref_ty.kind() {
            ty::Ref(_, _, mutbl) => mutbl,
            _ => unreachable!(),
        };
        let method_name = match mutbl {
            hir::Mutability::Not => "iter",
            hir::Mutability::Mut => "iter_mut",
        };
        (ty_name, method_name)
    })
}
