use super::TRANSMUTE_INT_TO_CHAR;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{std_or_core, sugg};
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_int_to_char` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    const_context: bool,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Int(ty::IntTy::I32) | ty::Uint(ty::UintTy::U32), &ty::Char) if !const_context => {
            span_lint_and_then(
                cx,
                TRANSMUTE_INT_TO_CHAR,
                e.span,
                format!("transmute from a `{from_ty}` to a `char`"),
                |diag| {
                    let Some(top_crate) = std_or_core(cx) else { return };
                    let arg = sugg::Sugg::hir(cx, arg, "..");
                    let arg = if let ty::Int(_) = from_ty.kind() {
                        arg.as_ty(ast::UintTy::U32.name_str())
                    } else {
                        arg
                    };
                    diag.span_suggestion(
                        e.span,
                        "consider using",
                        format!("{top_crate}::char::from_u32({arg}).unwrap()"),
                        Applicability::Unspecified,
                    );
                },
            );
            true
        },
        _ => false,
    }
}
