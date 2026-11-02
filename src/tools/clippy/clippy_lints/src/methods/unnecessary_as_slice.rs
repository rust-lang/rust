use super::UNNECESSARY_AS_SLICE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{get_parent_expr, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_span::Symbol;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, name: Symbol) {
    if !expr.span.from_expansion()
        && let Some(parent_expr) = get_parent_expr(cx, expr)
        // the parent is also a method call
        && let ExprKind::MethodCall(PathSegment { ident: parent_method_ident, .. }, parent_recv, ..) = parent_expr.kind
        // Excluding `as_ptr` and `as_mut_ptr` because they are commonly used to get raw pointers for FFI or unsafe code, and removing them might change the semantics of the code.
        && !(parent_method_ident.name == sym::as_ptr || parent_method_ident.name == sym::as_mut_ptr)
        // the parent's recv is the same as the current expr
        && parent_recv.hir_id == expr.hir_id
        // the parent is not a trait method
        && let opt_fn_id = cx.typeck_results().type_dependent_def_id(parent_expr.hir_id)
        && opt_fn_id.is_none_or(|fn_id| cx.tcx.trait_of_assoc(fn_id).is_none())
        // checking whether the recv is a `Vec`
        && cx
            .typeck_results()
            .expr_ty(recv)
            .peel_refs()
            .ty_adt_def()
            .is_some_and(|adt| cx.tcx.is_diagnostic_item(sym::Vec, adt.did()))
        && let Some(as_slice_span) = expr.span.trim_start(recv.span)
    {
        let parent_method_name = parent_method_ident.name;
        span_lint_and_sugg(
            cx,
            UNNECESSARY_AS_SLICE,
            as_slice_span,
            format!(
                "this `{name}` is unnecessary because the following method (`{parent_method_name}`) can be called on `Vec` directly"
            ),
            format!("remove `{name}`"),
            String::new(),
            Applicability::MachineApplicable,
        );
    }
}
