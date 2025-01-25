use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

pub fn check<'tcx>(cx: &LateContext<'tcx>, ty_into: Ty<'_>, cast_to_hir: &'tcx rustc_hir::Ty<'tcx>) {
    if let rustc_hir::TyKind::Ptr(rustc_hir::MutTy { ty, .. }) = cast_to_hir.kind
        && matches!(ty.kind, rustc_hir::TyKind::Infer(()))
    {
        clippy_utils::diagnostics::span_lint_and_sugg(
            cx,
            super::AS_POINTER_UNDERSCORE,
            cast_to_hir.span,
            "using inferred pointer cast",
            "use explicit type",
            ty_into.to_string(),
            Applicability::MachineApplicable,
        );
    }
}
