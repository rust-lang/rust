use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_no_std_crate;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, TypeAndMut};

use super::REF_AS_PTR;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_to_hir_ty: &Ty<'_>) {
    let (cast_from, cast_to) = (
        cx.typeck_results().expr_ty(cast_expr),
        cx.typeck_results().expr_ty(expr),
    );

    if matches!(cast_from.kind(), ty::Ref(..))
        && let ty::RawPtr(TypeAndMut { mutbl: to_mutbl, .. }) = cast_to.kind()
    {
        let core_or_std = if is_no_std_crate(cx) { "core" } else { "std" };
        let fn_name = match to_mutbl {
            Mutability::Not => "from_ref",
            Mutability::Mut => "from_mut",
        };

        let mut app = Applicability::MachineApplicable;
        let turbofish = match &cast_to_hir_ty.kind {
            TyKind::Infer => String::new(),
            TyKind::Ptr(mut_ty) => {
                if matches!(mut_ty.ty.kind, TyKind::Infer) {
                    String::new()
                } else {
                    format!(
                        "::<{}>",
                        snippet_with_applicability(cx, mut_ty.ty.span, "/* type */", &mut app)
                    )
                }
            },
            _ => return,
        };

        let cast_expr_sugg = Sugg::hir_with_applicability(cx, cast_expr, "_", &mut app);

        span_lint_and_sugg(
            cx,
            REF_AS_PTR,
            expr.span,
            "reference as raw pointer",
            "try",
            format!("{core_or_std}::ptr::{fn_name}{turbofish}({cast_expr_sugg})"),
            app,
        );
    }
}
