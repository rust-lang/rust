//! See [`complete_fn_param`].

use hir::HirDisplay;
use rustc_hash::FxHashMap;
use syntax::{
    algo,
    ast::{self, HasModuleItem},
    match_ast, AstNode, Direction, SyntaxKind,
};

use crate::{
    context::{ParamKind, PatternContext},
    CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
///
/// Also complete parameters for closure or local functions from the surrounding defined locals.
pub(crate) fn complete_fn_param(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let (param_list, _, param_kind) = match &ctx.pattern_ctx {
        Some(PatternContext { param_ctx: Some(kind), .. }) => kind,
        _ => return None,
    };

    let comma_wrapper = comma_wrapper(ctx);
    let mut add_new_item_to_acc = |label: &str, lookup: String| {
        let mk_item = |label: &str| {
            CompletionItem::new(CompletionItemKind::Binding, ctx.source_range(), label)
        };
        let mut item = match &comma_wrapper {
            Some(fmt) => mk_item(&fmt(label)),
            None => mk_item(label),
        };
        item.lookup_by(lookup);
        item.add_to(acc)
    };

    match param_kind {
        ParamKind::Function(function) => {
            fill_fn_params(ctx, function, param_list, add_new_item_to_acc);
        }
        ParamKind::Closure(closure) => {
            let stmt_list = closure.syntax().ancestors().find_map(ast::StmtList::cast)?;
            params_from_stmt_list_scope(ctx, stmt_list, |name, ty| {
                add_new_item_to_acc(&format!("{name}: {ty}"), name.to_string());
            });
        }
    }

    Some(())
}

fn fill_fn_params(
    ctx: &CompletionContext,
    function: &ast::Fn,
    param_list: &ast::ParamList,
    mut add_new_item_to_acc: impl FnMut(&str, String),
) {
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

    if let Some(stmt_list) = function.syntax().parent().and_then(ast::StmtList::cast) {
        params_from_stmt_list_scope(ctx, stmt_list, |name, ty| {
            file_params.entry(format!("{name}: {ty}")).or_insert(name.to_string());
        });
    }

    remove_duplicated(&mut file_params, param_list.params());
    let self_completion_items = ["self", "&self", "mut self", "&mut self"];
    if should_add_self_completions(ctx, param_list) {
        self_completion_items
            .into_iter()
            .for_each(|self_item| add_new_item_to_acc(self_item, self_item.to_string()));
    }

    file_params
        .into_iter()
        .for_each(|(whole_param, binding)| add_new_item_to_acc(&whole_param, binding));
}

fn params_from_stmt_list_scope(
    ctx: &CompletionContext,
    stmt_list: ast::StmtList,
    mut cb: impl FnMut(hir::Name, String),
) {
    let syntax_node = match stmt_list.syntax().last_child() {
        Some(it) => it,
        None => return,
    };
    let scope = ctx.sema.scope_at_offset(stmt_list.syntax(), syntax_node.text_range().end());
    let module = match scope.module() {
        Some(it) => it,
        None => return,
    };
    scope.process_all_names(&mut |name, def| {
        if let hir::ScopeDef::Local(local) = def {
            if let Ok(ty) = local.ty(ctx.db).display_source_code(ctx.db, module.into()) {
                cb(name, ty);
            }
        }
    });
}

fn remove_duplicated(
    file_params: &mut FxHashMap<String, String>,
    fn_params: ast::AstChildren<ast::Param>,
) {
    fn_params.for_each(|param| {
        let whole_param = param.syntax().text().to_string();
        file_params.remove(&whole_param);

        match param.pat() {
            // remove suggestions for patterns that already exist
            // if the type is missing we are checking the current param to be completed
            // in which case this would find itself removing the suggestions due to itself
            Some(pattern) if param.ty().is_some() => {
                let binding = pattern.syntax().text().to_string();
                file_params.retain(|_, v| v != &binding);
            }
            _ => (),
        }
    })
}

fn should_add_self_completions(ctx: &CompletionContext, param_list: &ast::ParamList) -> bool {
    let inside_impl = ctx.impl_def.is_some();
    let no_params = param_list.params().next().is_none() && param_list.self_param().is_none();

    inside_impl && no_params
}

fn comma_wrapper(ctx: &CompletionContext) -> Option<impl Fn(&str) -> String> {
    let next_token_kind = {
        let t = ctx.token.next_token()?;
        let t = algo::skip_whitespace_token(t, Direction::Next)?;
        t.kind()
    };
    let prev_token_kind = {
        let t = ctx.previous_token.clone()?;
        let t = algo::skip_whitespace_token(t, Direction::Prev)?;
        t.kind()
    };

    let has_trailing_comma =
        matches!(next_token_kind, SyntaxKind::COMMA | SyntaxKind::R_PAREN | SyntaxKind::PIPE);
    let trailing = if has_trailing_comma { "" } else { "," };

    let has_leading_comma =
        matches!(prev_token_kind, SyntaxKind::COMMA | SyntaxKind::L_PAREN | SyntaxKind::PIPE);
    let leading = if has_leading_comma { "" } else { ", " };

    Some(move |param: &_| format!("{}{}{}", leading, param, trailing))
}
