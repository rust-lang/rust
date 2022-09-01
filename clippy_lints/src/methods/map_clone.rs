use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item};
use clippy_utils::{is_diag_trait_item, meets_msrv, msrvs, peel_blocks};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::mir::Mutability;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_semver::RustcVersion;
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span};

use super::MAP_CLONE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    e: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    arg: &'tcx hir::Expr<'_>,
    msrv: Option<RustcVersion>,
) {
    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(e.hir_id);
        if cx.tcx.impl_of_method(method_id)
            .map_or(false, |id| is_type_diagnostic_item(cx, cx.tcx.type_of(id), sym::Option))
            || is_diag_trait_item(cx, method_id, sym::Iterator);
        if let hir::ExprKind::Closure(&hir::Closure{ body, .. }) = arg.kind;
        then {
            let closure_body = cx.tcx.hir().body(body);
            let closure_expr = peel_blocks(&closure_body.value);
            match closure_body.params[0].pat.kind {
                hir::PatKind::Ref(inner, hir::Mutability::Not) => if let hir::PatKind::Binding(
                    hir::BindingAnnotation::Unannotated, .., name, None
                ) = inner.kind {
                    if ident_eq(name, closure_expr) {
                        lint_explicit_closure(cx, e.span, recv.span, true, msrv);
                    }
                },
                hir::PatKind::Binding(hir::BindingAnnotation::Unannotated, .., name, None) => {
                    match closure_expr.kind {
                        hir::ExprKind::Unary(hir::UnOp::Deref, inner) => {
                            if ident_eq(name, inner) {
                                if let ty::Ref(.., Mutability::Not) = cx.typeck_results().expr_ty(inner).kind() {
                                    lint_explicit_closure(cx, e.span, recv.span, true, msrv);
                                }
                            }
                        },
                        hir::ExprKind::MethodCall(method, obj, [], _) => if_chain! {
                            if ident_eq(name, obj) && method.ident.name == sym::clone;
                            if let Some(fn_id) = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id);
                            if let Some(trait_id) = cx.tcx.trait_of_item(fn_id);
                            if cx.tcx.lang_items().clone_trait().map_or(false, |id| id == trait_id);
                            // no autoderefs
                            if !cx.typeck_results().expr_adjustments(obj).iter()
                                .any(|a| matches!(a.kind, Adjust::Deref(Some(..))));
                            then {
                                let obj_ty = cx.typeck_results().expr_ty(obj);
                                if let ty::Ref(_, ty, mutability) = obj_ty.kind() {
                                    if matches!(mutability, Mutability::Not) {
                                        let copy = is_copy(cx, *ty);
                                        lint_explicit_closure(cx, e.span, recv.span, copy, msrv);
                                    }
                                } else {
                                    lint_needless_cloning(cx, e.span, recv.span);
                                }
                            }
                        },
                        _ => {},
                    }
                },
                _ => {},
            }
        }
    }
}

fn ident_eq(name: Ident, path: &hir::Expr<'_>) -> bool {
    if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = path.kind {
        path.segments.len() == 1 && path.segments[0].ident == name
    } else {
        false
    }
}

fn lint_needless_cloning(cx: &LateContext<'_>, root: Span, receiver: Span) {
    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        root.trim_start(receiver).unwrap(),
        "you are needlessly cloning iterator elements",
        "remove the `map` call",
        String::new(),
        Applicability::MachineApplicable,
    );
}

fn lint_explicit_closure(cx: &LateContext<'_>, replace: Span, root: Span, is_copy: bool, msrv: Option<RustcVersion>) {
    let mut applicability = Applicability::MachineApplicable;

    let (message, sugg_method) = if is_copy && meets_msrv(msrv, msrvs::ITERATOR_COPIED) {
        ("you are using an explicit closure for copying elements", "copied")
    } else {
        ("you are using an explicit closure for cloning elements", "cloned")
    };

    span_lint_and_sugg(
        cx,
        MAP_CLONE,
        replace,
        message,
        &format!("consider calling the dedicated `{}` method", sugg_method),
        format!(
            "{}.{}()",
            snippet_with_applicability(cx, root, "..", &mut applicability),
            sugg_method,
        ),
        applicability,
    );
}
