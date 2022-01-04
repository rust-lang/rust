//! See [`complete_fn_param`].

use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, HasModuleItem},
    match_ast, AstNode, SyntaxKind,
};

use crate::{
    context::{ParamKind, PatternContext},
    CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
pub(crate) fn complete_fn_param(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let param_of_fn =
        matches!(ctx.pattern_ctx, Some(PatternContext { is_param: Some(ParamKind::Function), .. }));

    if !param_of_fn {
        return None;
    }

    let mut file_params = FxHashMap::default();

    let mut extract_params = |f: ast::Fn| {
        f.param_list().into_iter().flat_map(|it| it.params()).for_each(|param| {
            if let Some(pat) = param.pat() {
                // FIXME: We should be able to turn these into SmolStr without having to allocate a String
                let whole_param = param.syntax().text().to_string();
                let binding = pat.syntax().text().to_string();
                file_params.entry(whole_param).or_insert(binding);
            }
        });
    };

    for node in ctx.token.ancestors() {
        match_ast! {
            match node {
                ast::SourceFile(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut extract_params),
                ast::ItemList(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut extract_params),
                ast::AssocItemList(it) => it.assoc_items().filter_map(|item| match item {
                    ast::AssocItem::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut extract_params),
                _ => continue,
            }
        };
    }

    let function = ctx.token.ancestors().find_map(ast::Fn::cast)?;
    let param_list = function.param_list()?;

    remove_duplicated(&mut file_params, param_list.params());

    let self_completion_items = ["self", "&self", "mut self", "&mut self"];
    if should_add_self_completions(ctx, param_list) {
        self_completion_items.into_iter().for_each(|self_item| {
            add_new_item_to_acc(ctx, acc, self_item.to_string(), self_item.to_string())
        });
    }

    file_params.into_iter().try_for_each(|(whole_param, binding)| {
        Some(add_new_item_to_acc(ctx, acc, surround_with_commas(ctx, whole_param), binding))
    })?;

    Some(())
}

fn remove_duplicated(
    file_params: &mut FxHashMap<String, String>,
    fn_params: ast::AstChildren<ast::Param>,
) {
    fn_params.for_each(|param| {
        let whole_param = param.syntax().text().to_string();
        file_params.remove(&whole_param);

        if let Some(pattern) = param.pat() {
            let binding = pattern.syntax().text().to_string();
            file_params.retain(|_, v| v != &binding);
        }
    })
}

fn should_add_self_completions(ctx: &CompletionContext, param_list: ast::ParamList) -> bool {
    let inside_impl = ctx.impl_def.is_some();
    let no_params = param_list.params().next().is_none() && param_list.self_param().is_none();

    inside_impl && no_params
}

fn surround_with_commas(ctx: &CompletionContext, param: String) -> String {
    match fallible_surround_with_commas(ctx, &param) {
        Some(surrounded) => surrounded,
        // fallback to the original parameter
        None => param,
    }
}

fn fallible_surround_with_commas(ctx: &CompletionContext, param: &str) -> Option<String> {
    let next_token = {
        let t = ctx.token.next_token()?;
        match t.kind() {
            SyntaxKind::WHITESPACE => t.next_token()?,
            _ => t,
        }
    };

    let trailing_comma_missing = matches!(next_token.kind(), SyntaxKind::IDENT);
    let trailing = if trailing_comma_missing { "," } else { "" };

    let previous_token = if matches!(ctx.token.kind(), SyntaxKind::IDENT | SyntaxKind::WHITESPACE) {
        ctx.previous_token.as_ref()?
    } else {
        &ctx.token
    };

    let needs_leading = !matches!(previous_token.kind(), SyntaxKind::L_PAREN | SyntaxKind::COMMA);
    let leading = if needs_leading { ", " } else { "" };

    Some(format!("{}{}{}", leading, param, trailing))
}

fn add_new_item_to_acc(
    ctx: &CompletionContext,
    acc: &mut Completions,
    label: String,
    lookup: String,
) {
    let mut item = CompletionItem::new(CompletionItemKind::Binding, ctx.source_range(), label);
    item.lookup_by(lookup);
    item.add_to(acc)
}
