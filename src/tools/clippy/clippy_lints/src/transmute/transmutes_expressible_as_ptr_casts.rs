use super::utils::check_cast;
use super::TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use rustc_ast::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::cast::CastKind;
use rustc_middle::ty::Ty;

/// Checks for `transmutes_expressible_as_ptr_casts` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    from_ty_adjusted: bool,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
) -> bool {
    use CastKind::{AddrPtrCast, ArrayPtrCast, FnPtrAddrCast, FnPtrPtrCast, PtrAddrCast, PtrPtrCast};
    let mut app = Applicability::MachineApplicable;
    let mut sugg = match check_cast(cx, e, from_ty, to_ty) {
        Some(PtrPtrCast | AddrPtrCast | ArrayPtrCast | FnPtrPtrCast | FnPtrAddrCast) => {
            Sugg::hir_with_context(cx, arg, e.span.ctxt(), "..", &mut app)
                .as_ty(to_ty.to_string())
                .to_string()
        },
        Some(PtrAddrCast) if !from_ty_adjusted => Sugg::hir_with_context(cx, arg, e.span.ctxt(), "..", &mut app)
            .as_ty(to_ty.to_string())
            .to_string(),

        // The only adjustments here would be ref-to-ptr and unsize coercions. The result of an unsize coercions can't
        // be transmuted to a usize. For ref-to-ptr coercions, borrows need to be cast to a pointer before being cast to
        // a usize.
        Some(PtrAddrCast) => format!(
            "{} as {to_ty}",
            Sugg::hir_with_context(cx, arg, e.span.ctxt(), "..", &mut app).as_ty(from_ty)
        ),
        _ => return false,
    };

    if let Node::Expr(parent) = cx.tcx.hir().get_parent(e.hir_id)
        && parent.precedence().order() > ExprPrecedence::Cast.order()
    {
        sugg = format!("({sugg})");
    }

    span_lint_and_sugg(
        cx,
        TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS,
        e.span,
        &format!("transmute from `{from_ty}` to `{to_ty}` which could be expressed as a pointer cast instead"),
        "try",
        sugg,
        app,
    );
    true
}
