//! See [`complete_fn_param`].

use hir::HirDisplay;
use ide_db::FxHashMap;
use syntax::{
    algo,
    ast::{self, HasModuleItem},
    match_ast, AstNode, Direction, SyntaxKind, TextRange, TextSize,
};

use crate::{
    context::{ParamContext, ParamKind, PatternContext},
    CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

// FIXME: Make this a submodule of [`pattern`]
/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label will be suggested.
///
/// Also complete parameters for closure or local functions from the surrounding defined locals.
pub(crate) fn complete_fn_param(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
) -> Option<()> {
    let (ParamContext { param_list, kind, .. }, impl_) = match pattern_ctx {
        PatternContext { param_ctx: Some(kind), impl_, .. } => (kind, impl_),
        _ => return None,
    };

    let comma_wrapper = comma_wrapper(ctx);
    let mut add_new_item_to_acc = |label: &str| {
        let mk_item = |label: &str, range: TextRange| {
            CompletionItem::new(CompletionItemKind::Binding, range, label)
        };
        let item = match &comma_wrapper {
            Some((fmt, range)) => mk_item(&fmt(label), *range),
            None => mk_item(label, ctx.source_range()),
        };
        // Completion lookup is omitted intentionally here.
        // See the full discussion: https://github.com/rust-lang/rust-analyzer/issues/12073
        item.add_to(acc, ctx.db)
    };

    match kind {
        ParamKind::Function(function) => {
            fill_fn_params(ctx, function, param_list, impl_, add_new_item_to_acc);
        }
        ParamKind::Closure(closure) => {
            let stmt_list = closure.syntax().ancestors().find_map(ast::StmtList::cast)?;
            params_from_stmt_list_scope(ctx, stmt_list, |name, ty| {
                add_new_item_to_acc(&format!("{}: {ty}", name.display(ctx.db)));
            });
        }
    }

    Some(())
}

fn fill_fn_params(
    ctx: &CompletionContext<'_>,
    function: &ast::Fn,
    param_list: &ast::ParamList,
    impl_: &Option<ast::Impl>,
    mut add_new_item_to_acc: impl FnMut(&str),
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

    for node in ctx.token.parent_ancestors() {
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
            file_params
                .entry(format!("{}: {ty}", name.display(ctx.db)))
                .or_insert(name.display(ctx.db).to_string());
        });
    }
    remove_duplicated(&mut file_params, param_list.params());
    let self_completion_items = ["self", "&self", "mut self", "&mut self"];
    if should_add_self_completions(ctx.token.text_range().start(), param_list, impl_) {
        self_completion_items.into_iter().for_each(|self_item| add_new_item_to_acc(self_item));
    }

    file_params.keys().for_each(|whole_param| add_new_item_to_acc(whole_param));
}

fn params_from_stmt_list_scope(
    ctx: &CompletionContext<'_>,
    stmt_list: ast::StmtList,
    mut cb: impl FnMut(hir::Name, String),
) {
    let syntax_node = match stmt_list.syntax().last_child() {
        Some(it) => it,
        None => return,
    };
    if let Some(scope) =
        ctx.sema.scope_at_offset(stmt_list.syntax(), syntax_node.text_range().end())
    {
        let module = scope.module().into();
        scope.process_all_names(&mut |name, def| {
            if let hir::ScopeDef::Local(local) = def {
                if let Ok(ty) = local.ty(ctx.db).display_source_code(ctx.db, module, true) {
                    cb(name, ty);
                }
            }
        });
    }
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

fn should_add_self_completions(
    cursor: TextSize,
    param_list: &ast::ParamList,
    impl_: &Option<ast::Impl>,
) -> bool {
    if impl_.is_none() || param_list.self_param().is_some() {
        return false;
    }
    match param_list.params().next() {
        Some(first) => first.pat().map_or(false, |pat| pat.syntax().text_range().contains(cursor)),
        None => true,
    }
}

fn comma_wrapper(ctx: &CompletionContext<'_>) -> Option<(impl Fn(&str) -> String, TextRange)> {
    let param = ctx.token.parent_ancestors().find(|node| node.kind() == SyntaxKind::PARAM)?;

    let next_token_kind = {
        let t = param.last_token()?.next_token()?;
        let t = algo::skip_whitespace_token(t, Direction::Next)?;
        t.kind()
    };
    let prev_token_kind = {
        let t = param.first_token()?.prev_token()?;
        let t = algo::skip_whitespace_token(t, Direction::Prev)?;
        t.kind()
    };

    let has_trailing_comma =
        matches!(next_token_kind, SyntaxKind::COMMA | SyntaxKind::R_PAREN | SyntaxKind::PIPE);
    let trailing = if has_trailing_comma { "" } else { "," };

    let has_leading_comma =
        matches!(prev_token_kind, SyntaxKind::COMMA | SyntaxKind::L_PAREN | SyntaxKind::PIPE);
    let leading = if has_leading_comma { "" } else { ", " };

    Some((move |label: &_| (format!("{leading}{label}{trailing}")), param.text_range()))
}
