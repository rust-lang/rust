use crate::{Assist, AssistId, AssistCtx, ast_editor::{AstEditor, AstBuilder}};

use hir::db::HirDatabase;
use ra_syntax::{SmolStr, TreeArc};
use ra_syntax::ast::{self, AstNode, FnDef, ImplItem, ImplItemKind, NameOwner};
use ra_db::FilePosition;

enum AddMissingImplMembersMode {
    DefaultMethodsOnly,
    NoDefaultMethods,
}

pub(crate) fn add_missing_impl_members(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::NoDefaultMethods,
        "add_impl_missing_members",
        "add missing impl members",
    )
}

pub(crate) fn add_missing_default_members(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::DefaultMethodsOnly,
        "add_impl_default_members",
        "add impl default members",
    )
}

fn add_missing_impl_members_inner(
    mut ctx: AssistCtx<impl HirDatabase>,
    mode: AddMissingImplMembersMode,
    assist_id: &'static str,
    label: &'static str,
) -> Option<Assist> {
    let impl_node = ctx.node_at_offset::<ast::ImplBlock>()?;
    let impl_item_list = impl_node.item_list()?;

    let trait_def = {
        let file_id = ctx.frange.file_id;
        let position = FilePosition { file_id, offset: impl_node.syntax().range().start() };
        let analyzer = hir::SourceAnalyzer::new(ctx.db, position.file_id, impl_node.syntax(), None);

        resolve_target_trait_def(ctx.db, &analyzer, impl_node)?
    };

    let missing_fns: Vec<_> = {
        let fn_def_opt = |kind| if let ImplItemKind::FnDef(def) = kind { Some(def) } else { None };
        let def_name = |def| -> Option<&SmolStr> { FnDef::name(def).map(ast::Name::text) };

        let trait_items =
            trait_def.syntax().descendants().find_map(ast::ItemList::cast)?.impl_items();
        let impl_items = impl_item_list.impl_items();

        let trait_fns = trait_items.map(ImplItem::kind).filter_map(fn_def_opt);
        let impl_fns = impl_items.map(ImplItem::kind).filter_map(fn_def_opt).collect::<Vec<_>>();

        trait_fns
            .filter(|t| def_name(t).is_some())
            .filter(|t| match mode {
                AddMissingImplMembersMode::DefaultMethodsOnly => t.body().is_some(),
                AddMissingImplMembersMode::NoDefaultMethods => t.body().is_none(),
            })
            .filter(|t| impl_fns.iter().all(|i| def_name(i) != def_name(t)))
            .collect()
    };
    if missing_fns.is_empty() {
        return None;
    }

    ctx.add_action(AssistId(assist_id), label, |edit| {
        let n_existing_items = impl_item_list.impl_items().count();
        let fns = missing_fns.into_iter().map(add_body_and_strip_docstring).collect::<Vec<_>>();

        let mut ast_editor = AstEditor::new(impl_item_list);
        if n_existing_items == 0 {
            ast_editor.make_multiline();
        }
        ast_editor.append_functions(fns.iter().map(|it| &**it));
        let first_new_item = ast_editor.ast().impl_items().nth(n_existing_items).unwrap();
        let cursor_poisition = first_new_item.syntax().range().start();
        ast_editor.into_text_edit(edit.text_edit_builder());

        edit.set_cursor(cursor_poisition);
    });

    ctx.build()
}

fn add_body_and_strip_docstring(fn_def: &ast::FnDef) -> TreeArc<ast::FnDef> {
    let mut ast_editor = AstEditor::new(fn_def);
    if fn_def.body().is_none() {
        ast_editor.set_body(&AstBuilder::<ast::Block>::single_expr(
            &AstBuilder::<ast::Expr>::unimplemented(),
        ));
    }
    ast_editor.strip_attrs_and_docs();
    ast_editor.ast().to_owned()
}

/// Given an `ast::ImplBlock`, resolves the target trait (the one being
/// implemented) to a `ast::TraitDef`.
fn resolve_target_trait_def(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock,
) -> Option<TreeArc<ast::TraitDef>> {
    let ast_path =
        impl_block.target_trait().map(AstNode::syntax).and_then(ast::PathType::cast)?.path()?;

    match analyzer.resolve_path(db, &ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => Some(def.source(db).ast),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_add_missing_impl_members() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo {
    fn foo(&self);
    fn bar(&self);
    fn baz(&self);
}

struct S;

impl Foo for S {
    fn bar(&self) {}
<|>
}",
            "
trait Foo {
    fn foo(&self);
    fn bar(&self);
    fn baz(&self);
}

struct S;

impl Foo for S {
    fn bar(&self) {}
    <|>fn foo(&self) { unimplemented!() }
    fn baz(&self) { unimplemented!() }

}",
        );
    }

    #[test]
    fn test_copied_overriden_members() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo {
    fn foo(&self);
    fn bar(&self) -> bool { true }
    fn baz(&self) -> u32 { 42 }
}

struct S;

impl Foo for S {
    fn bar(&self) {}
<|>
}",
            "
trait Foo {
    fn foo(&self);
    fn bar(&self) -> bool { true }
    fn baz(&self) -> u32 { 42 }
}

struct S;

impl Foo for S {
    fn bar(&self) {}
    <|>fn foo(&self) { unimplemented!() }

}",
        );
    }

    #[test]
    fn test_empty_impl_block() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo { fn foo(&self); }
struct S;
impl Foo for S { <|> }",
            "
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {
    <|>fn foo(&self) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_cursor_after_empty_impl_block() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {}<|>",
            "
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {
    <|>fn foo(&self) { unimplemented!() }
}",
        )
    }

    #[test]
    fn test_empty_trait() {
        check_assist_not_applicable(
            add_missing_impl_members,
            "
trait Foo;
struct S;
impl Foo for S { <|> }",
        )
    }

    #[test]
    fn test_ignore_unnamed_trait_members_and_default_methods() {
        check_assist_not_applicable(
            add_missing_impl_members,
            "
trait Foo {
    fn (arg: u32);
    fn valid(some: u32) -> bool { false }
}
struct S;
impl Foo for S { <|> }",
        )
    }

    #[test]
    fn test_with_docstring_and_attrs() {
        check_assist(
            add_missing_impl_members,
            r#"
#[doc(alias = "test alias")]
trait Foo {
    /// doc string
    #[must_use]
    fn foo(&self);
}
struct S;
impl Foo for S {}<|>"#,
            r#"
#[doc(alias = "test alias")]
trait Foo {
    /// doc string
    #[must_use]
    fn foo(&self);
}
struct S;
impl Foo for S {
    <|>fn foo(&self) { unimplemented!() }
}"#,
        )
    }

    #[test]
    fn test_default_methods() {
        check_assist(
            add_missing_default_members,
            "
trait Foo {
    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S { <|> }",
            "
trait Foo {
    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S {
    <|>fn valid(some: u32) -> bool { false }
}",
        )
    }

}
