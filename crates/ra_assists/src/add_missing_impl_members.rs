use std::collections::HashSet;

use crate::assist_ctx::{Assist, AssistCtx};

use hir::Resolver;
use hir::db::HirDatabase;
use ra_syntax::{SmolStr, SyntaxKind, SyntaxNode, TreeArc};
use ra_syntax::ast::{self, AstNode, FnDef, ImplItem, ImplItemKind, NameOwner};
use ra_db::FilePosition;

/// Given an `ast::ImplBlock`, resolves the target trait (the one being
/// implemented) to a `ast::TraitDef`.
pub(crate) fn resolve_target_trait_def(
    db: &impl HirDatabase,
    resolver: &Resolver,
    impl_block: &ast::ImplBlock,
) -> Option<TreeArc<ast::TraitDef>> {
    let ast_path = impl_block.target_trait().map(AstNode::syntax).and_then(ast::PathType::cast)?;
    let hir_path = ast_path.path().and_then(hir::Path::from_ast)?;

    match resolver.resolve_path(db, &hir_path).take_types() {
        Some(hir::Resolution::Def(hir::ModuleDef::Trait(def))) => Some(def.source(db).1),
        _ => None,
    }
}

pub(crate) fn add_missing_impl_members(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    use SyntaxKind::{IMPL_BLOCK, ITEM_LIST, WHITESPACE};

    let node = ctx.covering_node();
    let kinds = node.ancestors().take(3).map(SyntaxNode::kind);
    // Only suggest this in `impl Foo for S { [Item...] <|> }` cursor position
    if !Iterator::eq(kinds, [WHITESPACE, ITEM_LIST, IMPL_BLOCK].iter().cloned()) {
        return None;
    }

    let impl_node = node.ancestors().find_map(ast::ImplBlock::cast)?;

    let trait_def = {
        let db = ctx.db;
        // TODO: Can we get the position of cursor itself rather than supplied range?
        let range = ctx.frange;
        let position = FilePosition { file_id: range.file_id, offset: range.range.start() };
        let resolver = hir::source_binder::resolver_for_position(db, position);

        resolve_target_trait_def(db, &resolver, impl_node)?
    };

    let fn_def_opt = |kind| if let ImplItemKind::FnDef(def) = kind { Some(def) } else { None };
    let def_name = |&def| -> Option<&SmolStr> { FnDef::name(def).map(ast::Name::text) };

    let trait_items = trait_def.syntax().descendants().find_map(ast::ItemList::cast)?.impl_items();
    let impl_items = impl_node.item_list()?.impl_items();

    let trait_fns = trait_items.map(ImplItem::kind).filter_map(fn_def_opt).collect::<Vec<_>>();
    let impl_fns = impl_items.map(ImplItem::kind).filter_map(fn_def_opt).collect::<Vec<_>>();

    let trait_fn_names = trait_fns.iter().filter_map(def_name).collect::<HashSet<_>>();
    let impl_fn_names = impl_fns.iter().filter_map(def_name).collect::<HashSet<_>>();

    let missing_fn_names = trait_fn_names.difference(&impl_fn_names).collect::<HashSet<_>>();
    let missing_fns = trait_fns
        .iter()
        .filter(|&t| def_name(t).map(|n| missing_fn_names.contains(&n)).unwrap_or(false));

    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{ check_assist };

    #[test]
    fn test_add_missing_impl_members() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo {
    fn foo(&self);
}

struct S;

impl Foo for S {
    <|>
}",
            "
trait Foo {
    fn foo(&self);
}

struct S;

impl Foo for S {
    fn foo(&self) {
        <|>
    }
}",
        );
    }
}
