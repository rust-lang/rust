use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{implements_trait, should_call_clone_as_function, walk_ptrs_ty_depth};
use clippy_utils::{get_parent_expr, is_diag_trait_item, path_to_local_id, peel_blocks, strip_pat_refs};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, LangItem};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_span::{Span, Symbol, sym};

use core::ops::ControlFlow;

use super::USELESS_ASREF;

/// Returns the first type inside the `Option`/`Result` type passed as argument.
fn get_enum_ty(enum_ty: Ty<'_>) -> Option<Ty<'_>> {
    struct ContainsTyVisitor {
        level: usize,
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsTyVisitor {
        type Result = ControlFlow<Ty<'tcx>>;

        fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
            self.level += 1;
            if self.level == 1 {
                t.super_visit_with(self)
            } else {
                ControlFlow::Break(t)
            }
        }
    }

    match enum_ty.visit_with(&mut ContainsTyVisitor { level: 0 }) {
        ControlFlow::Break(ty) => Some(ty),
        ControlFlow::Continue(()) => None,
    }
}

/// Checks for the `USELESS_ASREF` lint.
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, call_name: Symbol, recvr: &hir::Expr<'_>) {
    // when we get here, we've already checked that the call name is "as_ref" or "as_mut"
    // check if the call is to the actual `AsRef` or `AsMut` trait
    let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) else {
        return;
    };

    if is_diag_trait_item(cx, def_id, sym::AsRef) || is_diag_trait_item(cx, def_id, sym::AsMut) {
        // check if the type after `as_ref` or `as_mut` is the same as before
        let rcv_ty = cx.typeck_results().expr_ty(recvr);
        let res_ty = cx.typeck_results().expr_ty(expr);
        let (base_res_ty, res_depth) = walk_ptrs_ty_depth(res_ty);
        let (base_rcv_ty, rcv_depth) = walk_ptrs_ty_depth(rcv_ty);
        if base_rcv_ty == base_res_ty && rcv_depth >= res_depth {
            if let Some(parent) = get_parent_expr(cx, expr) {
                // allow the `as_ref` or `as_mut` if it is followed by another method call
                if let hir::ExprKind::MethodCall(segment, ..) = parent.kind
                    && segment.ident.span != expr.span
                {
                    return;
                }

                // allow the `as_ref` or `as_mut` if they belong to a closure that changes
                // the number of references
                if matches!(parent.kind, hir::ExprKind::Closure(..)) && rcv_depth != res_depth {
                    return;
                }
            }

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                USELESS_ASREF,
                expr.span,
                format!("this call to `{call_name}` does nothing"),
                "try",
                snippet_with_applicability(cx, recvr.span, "..", &mut applicability).to_string(),
                applicability,
            );
        }
    } else if let Some(impl_id) = cx.tcx.impl_of_method(def_id)
        && let Some(adt) = cx.tcx.type_of(impl_id).instantiate_identity().ty_adt_def()
        && matches!(cx.tcx.get_diagnostic_name(adt.did()), Some(sym::Option | sym::Result))
    {
        let rcv_ty = cx.typeck_results().expr_ty(recvr).peel_refs();
        let res_ty = cx.typeck_results().expr_ty(expr).peel_refs();

        if let Some(rcv_ty) = get_enum_ty(rcv_ty)
            && let Some(res_ty) = get_enum_ty(res_ty)
            // If the only thing the `as_mut`/`as_ref` call is doing is adding references and not
            // changing the type, then we can move forward.
            && rcv_ty.peel_refs() == res_ty.peel_refs()
            && let Some(parent) = get_parent_expr(cx, expr)
            // Check that it only has one argument.
            && let hir::ExprKind::MethodCall(segment, _, [arg], _) = parent.kind
            && segment.ident.span != expr.span
            // We check that the called method name is `map`.
            && segment.ident.name == sym::map
            && is_calling_clone(cx, arg)
            // And that we are not recommending recv.clone() over Arc::clone() or similar
            && !should_call_clone_as_function(cx, rcv_ty)
            // https://github.com/rust-lang/rust-clippy/issues/12357
            && let Some(clone_trait) = cx.tcx.lang_items().clone_trait()
            && implements_trait(cx, cx.typeck_results().expr_ty(recvr), clone_trait, &[])
        {
            lint_as_ref_clone(cx, expr.span.with_hi(parent.span.hi()), recvr, call_name);
        }
    }
}

fn check_qpath(cx: &LateContext<'_>, qpath: hir::QPath<'_>, hir_id: hir::HirId) -> bool {
    // We check it's calling the `clone` method of the `Clone` trait.
    if let Some(path_def_id) = cx.qpath_res(&qpath, hir_id).opt_def_id() {
        cx.tcx.lang_items().get(LangItem::CloneFn) == Some(path_def_id)
    } else {
        false
    }
}

fn is_calling_clone(cx: &LateContext<'_>, arg: &hir::Expr<'_>) -> bool {
    match arg.kind {
        hir::ExprKind::Closure(&hir::Closure { body, .. })
            // If it's a closure, we need to check what is called.
            if let closure_body = cx.tcx.hir_body(body)
                && let [param] = closure_body.params
                && let hir::PatKind::Binding(_, local_id, ..) = strip_pat_refs(param.pat).kind =>
        {
            let closure_expr = peel_blocks(closure_body.value);
            match closure_expr.kind {
                hir::ExprKind::MethodCall(method, obj, [], _) => {
                    if method.ident.name == sym::clone
                        && let Some(fn_id) = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id)
                        && let Some(trait_id) = cx.tcx.trait_of_item(fn_id)
                        // We check it's the `Clone` trait.
                        && cx.tcx.lang_items().clone_trait().is_some_and(|id| id == trait_id)
                        // no autoderefs
                        && !cx.typeck_results().expr_adjustments(obj).iter()
                            .any(|a| matches!(a.kind, Adjust::Deref(Some(..))))
                        && path_to_local_id(obj, local_id)
                    {
                        true
                    } else {
                        false
                    }
                },
                hir::ExprKind::Call(call, [recv]) => {
                    if let hir::ExprKind::Path(qpath) = call.kind
                        && path_to_local_id(recv, local_id)
                    {
                        check_qpath(cx, qpath, call.hir_id)
                    } else {
                        false
                    }
                },
                _ => false,
            }
        },
        hir::ExprKind::Path(qpath) => check_qpath(cx, qpath, arg.hir_id),
        _ => false,
    }
}

fn lint_as_ref_clone(cx: &LateContext<'_>, span: Span, recvr: &hir::Expr<'_>, call_name: Symbol) {
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        USELESS_ASREF,
        span,
        format!("this call to `{call_name}.map(...)` does nothing"),
        "try",
        format!(
            "{}.clone()",
            snippet_with_applicability(cx, recvr.span, "..", &mut applicability)
        ),
        applicability,
    );
}
