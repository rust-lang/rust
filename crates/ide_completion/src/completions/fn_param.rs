//! See `complete_fn_param`.

use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, ModuleItemOwner},
    match_ast, AstNode,
};

use crate::{
    context::{ParamKind, PatternContext},
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
pub(crate) fn complete_fn_param(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !matches!(ctx.pattern_ctx, Some(PatternContext { is_param: Some(ParamKind::Function), .. }))
    {
        return None;
    }

    let mut params = FxHashMap::default();

    let me = ctx.token.ancestors().find_map(ast::Fn::cast);
    let mut process_fn = |func: ast::Fn| {
        if Some(&func) == me.as_ref() {
            return;
        }
        func.param_list().into_iter().flat_map(|it| it.params()).for_each(|param| {
            if let Some(pat) = param.pat() {
                let text = param.syntax().text().to_string();
                let lookup = pat.syntax().text().to_string();
                params.entry(text).or_insert(lookup);
            }
        });
    };

    for node in ctx.token.ancestors() {
        match_ast! {
            match node {
                ast::SourceFile(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                ast::ItemList(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                ast::AssocItemList(it) => it.assoc_items().filter_map(|item| match item {
                    ast::AssocItem::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                _ => continue,
            }
        };
    }

    let self_completion_items = ["self", "&self", "mut self", "&mut self"];
    if ctx.impl_def.is_some() && me?.param_list()?.params().next().is_none() {
        self_completion_items.iter().for_each(|self_item| {
            add_new_item_to_acc(ctx, acc, self_item.to_string(), self_item.to_string())
        });
    }

    params.into_iter().for_each(|(label, lookup)| add_new_item_to_acc(ctx, acc, label, lookup));

    Some(())
}

fn add_new_item_to_acc(
    ctx: &CompletionContext,
    acc: &mut Completions,
    label: String,
    lookup: String,
) {
    let mut item = CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label);
    item.kind(CompletionItemKind::Binding).lookup_by(lookup);
    item.add_to(acc)
}
