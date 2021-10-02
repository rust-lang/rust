use rustc_hir as hir;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_span::{sym, Span};
use rustc_typeck::hir_ty_to_ty;

use if_chain::if_chain;

use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::trait_ref_of_method;
use clippy_utils::ty::is_type_diagnostic_item;

use super::RESULT_UNIT_ERR;

pub(super) fn check_item(cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
    if let hir::ItemKind::Fn(ref sig, ref _generics, _) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if is_public {
            check_result_unit_err(cx, sig.decl, item.span, fn_header_span);
        }
    }
}

pub(super) fn check_impl_item(cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
    if let hir::ImplItemKind::Fn(ref sig, _) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if is_public && trait_ref_of_method(cx, item.hir_id()).is_none() {
            check_result_unit_err(cx, sig.decl, item.span, fn_header_span);
        }
    }
}

pub(super) fn check_trait_item(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, _) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if is_public {
            check_result_unit_err(cx, sig.decl, item.span, fn_header_span);
        }
    }
}

fn check_result_unit_err(cx: &LateContext<'_>, decl: &hir::FnDecl<'_>, item_span: Span, fn_header_span: Span) {
    if_chain! {
        if !in_external_macro(cx.sess(), item_span);
        if let hir::FnRetTy::Return(ty) = decl.output;
        let ty = hir_ty_to_ty(cx.tcx, ty);
        if is_type_diagnostic_item(cx, ty, sym::Result);
        if let ty::Adt(_, substs) = ty.kind();
        let err_ty = substs.type_at(1);
        if err_ty.is_unit();
        then {
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
}
