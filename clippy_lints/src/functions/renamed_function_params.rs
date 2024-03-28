use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::OwnerId;
use rustc_hir::{ImplItem, ImplItemKind, ItemKind, Node};
use rustc_lint::LateContext;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::Span;

use super::RENAMED_FUNCTION_PARAMS;

pub(super) fn check_impl_item(cx: &LateContext<'_>, item: &ImplItem<'_>) {
    if !item.span.from_expansion()
        && let ImplItemKind::Fn(_, body_id) = item.kind
        && let Some(did) = trait_item_def_id_of_impl(cx, item.owner_id)
    {
        let mut param_idents_iter = cx.tcx.hir().body_param_names(body_id);
        let mut default_param_idents_iter = cx.tcx.fn_arg_names(did).iter().copied();

        let renames = RenamedFnArgs::new(&mut default_param_idents_iter, &mut param_idents_iter);
        if !renames.0.is_empty() {
            let multi_span = renames.multi_span();
            let plural = if renames.0.len() == 1 { "" } else { "s" };
            span_lint_and_then(
                cx,
                RENAMED_FUNCTION_PARAMS,
                multi_span,
                &format!("renamed function parameter{plural} of trait impl"),
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
    fn new<I, T>(default_names: &mut I, current_names: &mut T) -> Self
    where
        I: Iterator<Item = Ident>,
        T: Iterator<Item = Ident>,
    {
        let mut renamed: Vec<(Span, String)> = vec![];

        debug_assert!(default_names.size_hint() == current_names.size_hint());
        while let (Some(def_name), Some(cur_name)) = (default_names.next(), current_names.next()) {
            let current_name = cur_name.name;
            let default_name = def_name.name;
            if is_unused_or_empty_symbol(current_name) || is_unused_or_empty_symbol(default_name) {
                continue;
            }
            if current_name != default_name {
                renamed.push((cur_name.span, default_name.to_string()));
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

fn is_unused_or_empty_symbol(symbol: Symbol) -> bool {
    // FIXME: `body_param_names` currently returning empty symbols for `wild` as well,
    // so we need to check if the symbol is empty first.
    // Therefore the check of whether it's equal to [`kw::Underscore`] has no use for now,
    // but it would be nice to keep it here just to be future-proof.
    symbol.is_empty() || symbol == kw::Underscore || symbol.as_str().starts_with('_')
}

/// Get the [`trait_item_def_id`](rustc_hir::hir::ImplItemRef::trait_item_def_id) of an impl item.
fn trait_item_def_id_of_impl(cx: &LateContext<'_>, impl_item_id: OwnerId) -> Option<DefId> {
    let trait_node = cx.tcx.parent_hir_node(impl_item_id.into());
    if let Node::Item(item) = trait_node
        && let ItemKind::Impl(impl_) = &item.kind
    {
        impl_.items.iter().find_map(|item| {
            if item.id.owner_id == impl_item_id {
                item.trait_item_def_id
            } else {
                None
            }
        })
    } else {
        None
    }
}
