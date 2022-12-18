use rustc_errors::Diagnostic;
use rustc_hir as hir;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Adt, Ty};
use rustc_span::{sym, Span};

use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::trait_ref_of_method;
use clippy_utils::ty::{approx_ty_size, is_type_diagnostic_item, AdtVariantInfo};

use super::{RESULT_LARGE_ERR, RESULT_UNIT_ERR};

/// The type of the `Err`-variant in a `std::result::Result` returned by the
/// given `FnDecl`
fn result_err_ty<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &hir::FnDecl<'tcx>,
    id: hir::def_id::LocalDefId,
    item_span: Span,
) -> Option<(&'tcx hir::Ty<'tcx>, Ty<'tcx>)> {
    if !in_external_macro(cx.sess(), item_span)
        && let hir::FnRetTy::Return(hir_ty) = decl.output
        && let ty = cx.tcx.erase_late_bound_regions(cx.tcx.fn_sig(id).output())
        && is_type_diagnostic_item(cx, ty, sym::Result)
        && let ty::Adt(_, substs) = ty.kind()
    {
        let err_ty = substs.type_at(1);
        Some((hir_ty, err_ty))
    } else {
        None
    }
}

pub(super) fn check_item<'tcx>(cx: &LateContext<'tcx>, item: &hir::Item<'tcx>, large_err_threshold: u64) {
    if let hir::ItemKind::Fn(ref sig, _generics, _) = item.kind
        && let Some((hir_ty, err_ty)) = result_err_ty(cx, sig.decl, item.owner_id.def_id, item.span)
    {
        if cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
            check_result_unit_err(cx, err_ty, fn_header_span);
        }
        check_result_large_err(cx, err_ty, hir_ty.span, large_err_threshold);
    }
}

pub(super) fn check_impl_item<'tcx>(cx: &LateContext<'tcx>, item: &hir::ImplItem<'tcx>, large_err_threshold: u64) {
    // Don't lint if method is a trait's implementation, we can't do anything about those
    if let hir::ImplItemKind::Fn(ref sig, _) = item.kind
        && let Some((hir_ty, err_ty)) = result_err_ty(cx, sig.decl, item.owner_id.def_id, item.span)
        && trait_ref_of_method(cx, item.owner_id.def_id).is_none()
    {
        if cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
            check_result_unit_err(cx, err_ty, fn_header_span);
        }
        check_result_large_err(cx, err_ty, hir_ty.span, large_err_threshold);
    }
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &hir::TraitItem<'tcx>, large_err_threshold: u64) {
    if let hir::TraitItemKind::Fn(ref sig, _) = item.kind {
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if let Some((hir_ty, err_ty)) = result_err_ty(cx, sig.decl, item.owner_id.def_id, item.span) {
            if cx.effective_visibilities.is_exported(item.owner_id.def_id) {
                check_result_unit_err(cx, err_ty, fn_header_span);
            }
            check_result_large_err(cx, err_ty, hir_ty.span, large_err_threshold);
        }
    }
}

fn check_result_unit_err(cx: &LateContext<'_>, err_ty: Ty<'_>, fn_header_span: Span) {
    if err_ty.is_unit() {
        span_lint_and_help(
            cx,
            RESULT_UNIT_ERR,
            fn_header_span,
            "this returns a `Result<_, ()>`",
            None,
            "use a custom `Error` type instead",
        );
    }
}

fn check_result_large_err<'tcx>(cx: &LateContext<'tcx>, err_ty: Ty<'tcx>, hir_ty_span: Span, large_err_threshold: u64) {
    if_chain! {
        if let Adt(adt, subst) = err_ty.kind();
        if let Some(local_def_id) = err_ty.ty_adt_def().expect("already checked this is adt").did().as_local();
        if let Some(hir::Node::Item(item)) = cx
            .tcx
            .hir()
            .find_by_def_id(local_def_id);
        if let hir::ItemKind::Enum(ref def, _) = item.kind;
        then {
            let variants_size = AdtVariantInfo::new(cx, *adt, subst);
            if let Some((first_variant, variants)) = variants_size.split_first()
                && first_variant.size >= large_err_threshold
            {
                span_lint_and_then(
                    cx,
                    RESULT_LARGE_ERR,
                    hir_ty_span,
                    "the `Err`-variant returned from this function is very large",
                    |diag| {
                        diag.span_label(
                            def.variants[first_variant.ind].span,
                            format!("the largest variant contains at least {} bytes", variants_size[0].size),
                        );

                        for variant in variants {
                            if variant.size >= large_err_threshold {
                                let variant_def = &def.variants[variant.ind];
                                diag.span_label(
                                    variant_def.span,
                                    format!("the variant `{}` contains at least {} bytes", variant_def.ident, variant.size),
                                );
                            }
                        }

                        diag.help(format!("try reducing the size of `{err_ty}`, for example by boxing large elements or replacing it with `Box<{err_ty}>`"));
                    }
                );
            }
        }
        else {
            let ty_size = approx_ty_size(cx, err_ty);
            if ty_size >= large_err_threshold {
                span_lint_and_then(
                    cx,
                    RESULT_LARGE_ERR,
                    hir_ty_span,
                    "the `Err`-variant returned from this function is very large",
                    |diag: &mut Diagnostic| {
                        diag.span_label(hir_ty_span, format!("the `Err`-variant is at least {ty_size} bytes"));
                        diag.help(format!("try reducing the size of `{err_ty}`, for example by boxing large elements or replacing it with `Box<{err_ty}>`"));
                    },
                );
            }
        }
    }
}
