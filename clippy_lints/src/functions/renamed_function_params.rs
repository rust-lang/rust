use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_hir::hir_id::OwnerId;
use rustc_hir::{Impl, ImplItem, ImplItemKind, ImplItemRef, ItemKind, Node, TraitRef};
use rustc_lint::LateContext;
use rustc_span::Span;
use rustc_span::symbol::{Ident, kw};

use super::RENAMED_FUNCTION_PARAMS;

pub(super) fn check_impl_item(cx: &LateContext<'_>, item: &ImplItem<'_>, ignored_traits: &DefIdSet) {
    if !item.span.from_expansion()
        && let ImplItemKind::Fn(_, body_id) = item.kind
        && let parent_node = cx.tcx.parent_hir_node(item.hir_id())
        && let Node::Item(parent_item) = parent_node
        && let ItemKind::Impl(Impl {
            items,
            of_trait: Some(trait_ref),
            ..
        }) = &parent_item.kind
        && let Some(did) = trait_item_def_id_of_impl(items, item.owner_id)
        && !is_from_ignored_trait(trait_ref, ignored_traits)
    {
        let mut param_idents_iter = cx.tcx.hir_body_param_idents(body_id);
        let mut default_param_idents_iter = cx.tcx.fn_arg_idents(did).iter().copied();

        let renames = RenamedFnArgs::new(&mut default_param_idents_iter, &mut param_idents_iter);
        if !renames.0.is_empty() {
            let multi_span = renames.multi_span();
            let plural = if renames.0.len() == 1 { "" } else { "s" };
            span_lint_and_then(
                cx,
                RENAMED_FUNCTION_PARAMS,
                multi_span,
                format!("renamed function parameter{plural} of trait impl"),
                |diag| {
                    diag.multipart_suggestion(
                        format!("consider using the default name{plural}"),
                        renames.0,
                        Applicability::Unspecified,
                    );
                },
            );
        }
    }
}

struct RenamedFnArgs(Vec<(Span, String)>);

impl RenamedFnArgs {
    /// Comparing between an iterator of default names and one with current names,
    /// then collect the ones that got renamed.
    fn new<I1, I2>(default_idents: &mut I1, current_idents: &mut I2) -> Self
    where
        I1: Iterator<Item = Option<Ident>>,
        I2: Iterator<Item = Option<Ident>>,
    {
        let mut renamed: Vec<(Span, String)> = vec![];

        debug_assert!(default_idents.size_hint() == current_idents.size_hint());
        while let (Some(default_ident), Some(current_ident)) = (default_idents.next(), current_idents.next()) {
            let has_name_to_check = |ident: Option<Ident>| {
                if let Some(ident) = ident
                    && ident.name != kw::Underscore
                    && !ident.name.as_str().starts_with('_')
                {
                    Some(ident)
                } else {
                    None
                }
            };

            if let Some(default_ident) = has_name_to_check(default_ident)
                && let Some(current_ident) = has_name_to_check(current_ident)
                && default_ident.name != current_ident.name
            {
                renamed.push((current_ident.span, default_ident.to_string()));
            }
        }

        Self(renamed)
    }

    fn multi_span(&self) -> MultiSpan {
        self.0
            .iter()
            .map(|(span, _)| span)
            .copied()
            .collect::<Vec<Span>>()
            .into()
    }
}

/// Get the [`trait_item_def_id`](ImplItemRef::trait_item_def_id) of a relevant impl item.
fn trait_item_def_id_of_impl(items: &[ImplItemRef], target: OwnerId) -> Option<DefId> {
    items.iter().find_map(|item| {
        if item.id.owner_id == target {
            item.trait_item_def_id
        } else {
            None
        }
    })
}

fn is_from_ignored_trait(of_trait: &TraitRef<'_>, ignored_traits: &DefIdSet) -> bool {
    let Some(trait_did) = of_trait.trait_def_id() else {
        return false;
    };
    ignored_traits.contains(&trait_did)
}
