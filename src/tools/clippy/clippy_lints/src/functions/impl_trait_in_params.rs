use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_test;

use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, GenericParam, Generics, HirId, ImplItem, ImplItemKind, TraitItem, TraitItemKind};
use rustc_lint::LateContext;

use super::IMPL_TRAIT_IN_PARAMS;

fn report(cx: &LateContext<'_>, param: &GenericParam<'_>, generics: &Generics<'_>) {
    // No generics with nested generics, and no generics like FnMut(x)
    span_lint_and_then(
        cx,
        IMPL_TRAIT_IN_PARAMS,
        param.span,
        "`impl Trait` used as a function parameter",
        |diag| {
            if let Some(gen_span) = generics.span_for_param_suggestion() {
                // If there's already a generic param with the same bound, do not lint **this** suggestion.
                diag.span_suggestion_verbose(
                    gen_span,
                    "add a type parameter",
                    format!(", {{ /* Generic name */ }}: {}", &param.name.ident().as_str()[5..]),
                    Applicability::HasPlaceholders,
                );
            } else {
                diag.span_suggestion_verbose(
                    generics.span,
                    "add a type parameter",
                    format!("<{{ /* Generic name */ }}: {}>", &param.name.ident().as_str()[5..]),
                    Applicability::HasPlaceholders,
                );
            }
        },
    );
}

pub(super) fn check_fn<'tcx>(cx: &LateContext<'_>, kind: &'tcx FnKind<'_>, body: &'tcx Body<'_>, hir_id: HirId) {
    if let FnKind::ItemFn(_, generics, _) = kind
        && cx.tcx.visibility(cx.tcx.hir_body_owner_def_id(body.id())).is_public()
        && !is_in_test(cx.tcx, hir_id)
    {
        for param in generics.params {
            if param.is_impl_trait() {
                report(cx, param, generics);
            }
        }
    }
}

pub(super) fn check_impl_item(cx: &LateContext<'_>, impl_item: &ImplItem<'_>) {
    if let ImplItemKind::Fn(_, body_id) = impl_item.kind
        && let hir::Node::Item(item) = cx.tcx.parent_hir_node(impl_item.hir_id())
        && let hir::ItemKind::Impl(impl_) = item.kind
        && let hir::Impl { of_trait: None, .. } = impl_
        && let body = cx.tcx.hir_body(body_id)
        && cx.tcx.visibility(cx.tcx.hir_body_owner_def_id(body.id())).is_public()
        && !is_in_test(cx.tcx, impl_item.hir_id())
    {
        for param in impl_item.generics.params {
            if param.is_impl_trait() {
                report(cx, param, impl_item.generics);
            }
        }
    }
}

pub(super) fn check_trait_item(cx: &LateContext<'_>, trait_item: &TraitItem<'_>, avoid_breaking_exported_api: bool) {
    if !avoid_breaking_exported_api
        && let TraitItemKind::Fn(_, _) = trait_item.kind
        && let hir::Node::Item(item) = cx.tcx.parent_hir_node(trait_item.hir_id())
        // ^^ (Will always be a trait)
        && !item.vis_span.is_empty() // Is public
        && !is_in_test(cx.tcx, trait_item.hir_id())
    {
        for param in trait_item.generics.params {
            if param.is_impl_trait() {
                report(cx, param, trait_item.generics);
            }
        }
    }
}
