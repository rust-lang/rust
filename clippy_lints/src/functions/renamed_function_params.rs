use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::OwnerId;
use rustc_hir::{ImplItem, ImplItemKind, ItemKind, Node};
use rustc_lint::LateContext;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

use super::RENAMED_FUNCTION_PARAMS;

pub(super) fn check_impl_item(cx: &LateContext<'_>, item: &ImplItem<'_>) {
    if let ImplItemKind::Fn(_, body_id) = item.kind &&
        let Some(did) = impled_item_def_id(cx, item.owner_id)
    {
        let mut param_idents_iter = cx.tcx.hir().body_param_names(body_id);
        let mut default_param_idents_iter = cx.tcx.fn_arg_names(did).iter().copied();

        let renames = renamed_params(&mut default_param_idents_iter, &mut param_idents_iter);
        // FIXME: Should we use `MultiSpan` to combine output together?
        // But how should we display help message if so.
        for rename in renames {
            span_lint_and_help(
                cx,
                RENAMED_FUNCTION_PARAMS,
                rename.renamed_span,
                "function parameter name was renamed from its trait default",
                None,
                &format!("consider changing the name to: '{}'", rename.default_name.as_str())
            );
        }
    }
}

struct RenamedParam {
    renamed_span: Span,
    default_name: Symbol,
}

fn renamed_params<I, T>(default_names: &mut I, current_names: &mut T) -> Vec<RenamedParam>
where
    I: Iterator<Item = Ident>,
    T: Iterator<Item = Ident>,
{
    let mut renamed = vec![];
    // FIXME: Should we stop if they have different length?
    while let (Some(def_name), Some(cur_name)) = (default_names.next(), current_names.next()) {
        let current_name = cur_name.name;
        let default_name = def_name.name;
        if is_ignored_or_empty_symbol(current_name) || is_ignored_or_empty_symbol(default_name) {
            continue;
        }
        if current_name != default_name {
            renamed.push(RenamedParam {
                renamed_span: cur_name.span,
                default_name,
            });
        }
    }
    renamed
}

fn is_ignored_or_empty_symbol(symbol: Symbol) -> bool {
    let s = symbol.as_str();
    s.is_empty() || s.starts_with('_')
}

fn impled_item_def_id(cx: &LateContext<'_>, impl_item_id: OwnerId) -> Option<DefId> {
    let trait_node = cx.tcx.hir().find_parent(impl_item_id.into())?;
    if let Node::Item(item) = trait_node &&
        let ItemKind::Impl(impl_) = &item.kind
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
