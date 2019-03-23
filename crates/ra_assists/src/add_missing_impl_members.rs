use crate::{Assist, AssistId, AssistCtx};

use hir::Resolver;
use hir::db::HirDatabase;
use ra_syntax::{SmolStr, SyntaxKind, TextRange, TextUnit, TreeArc};
use ra_syntax::ast::{self, AstNode, FnDef, ImplItem, ImplItemKind, NameOwner};
use ra_db::FilePosition;
use ra_fmt::{leading_indent, reindent};

use itertools::Itertools;

pub(crate) fn add_missing_impl_members(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let impl_node = ctx.node_at_offset::<ast::ImplBlock>()?;
    let impl_item_list = impl_node.item_list()?;

    let trait_def = {
        let file_id = ctx.frange.file_id;
        let position = FilePosition { file_id, offset: impl_node.syntax().range().start() };
        let resolver = hir::source_binder::resolver_for_position(ctx.db, position);

        resolve_target_trait_def(ctx.db, &resolver, impl_node)?
    };

    let missing_fns: Vec<_> = {
        let fn_def_opt = |kind| if let ImplItemKind::FnDef(def) = kind { Some(def) } else { None };
        let def_name = |def| -> Option<&SmolStr> { FnDef::name(def).map(ast::Name::text) };

        let trait_items =
            trait_def.syntax().descendants().find_map(ast::ItemList::cast)?.impl_items();
        let impl_items = impl_item_list.impl_items();

        let trait_fns = trait_items.map(ImplItem::kind).filter_map(fn_def_opt).collect::<Vec<_>>();
        let impl_fns = impl_items.map(ImplItem::kind).filter_map(fn_def_opt).collect::<Vec<_>>();

        trait_fns
            .into_iter()
            .filter(|t| def_name(t).is_some())
            .filter(|t| t.body().is_none())
            .filter(|t| impl_fns.iter().all(|i| def_name(i) != def_name(t)))
            .collect()
    };
    if missing_fns.is_empty() {
        return None;
    }

    ctx.add_action(AssistId("add_impl_missing_members"), "add missing impl members", |edit| {
        let (parent_indent, indent) = {
            // FIXME: Find a way to get the indent already used in the file.
            // Now, we copy the indent of first item or indent with 4 spaces relative to impl block
            const DEFAULT_INDENT: &str = "    ";
            let first_item = impl_item_list.impl_items().next();
            let first_item_indent =
                first_item.and_then(|i| leading_indent(i.syntax())).map(ToOwned::to_owned);
            let impl_block_indent = leading_indent(impl_node.syntax()).unwrap_or_default();

            (
                impl_block_indent.to_owned(),
                first_item_indent.unwrap_or_else(|| impl_block_indent.to_owned() + DEFAULT_INDENT),
            )
        };

        let changed_range = {
            let children = impl_item_list.syntax().children();
            let last_whitespace = children.filter_map(ast::Whitespace::cast).last();

            last_whitespace.map(|w| w.syntax().range()).unwrap_or_else(|| {
                let in_brackets = impl_item_list.syntax().range().end() - TextUnit::of_str("}");
                TextRange::from_to(in_brackets, in_brackets)
            })
        };

        let func_bodies = format!("\n{}", missing_fns.into_iter().map(build_func_body).join("\n"));
        let trailing_whitespace = format!("\n{}", parent_indent);
        let func_bodies = reindent(&func_bodies, &indent) + &trailing_whitespace;

        let replaced_text_range = TextUnit::of_str(&func_bodies);

        edit.replace(changed_range, func_bodies);
        edit.set_cursor(
            changed_range.start() + replaced_text_range - TextUnit::of_str(&trailing_whitespace),
        );
    });

    ctx.build()
}

/// Given an `ast::ImplBlock`, resolves the target trait (the one being
/// implemented) to a `ast::TraitDef`.
fn resolve_target_trait_def(
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

fn build_func_body(def: &ast::FnDef) -> String {
    let mut buf = String::new();

    for child in def.syntax().children() {
        match (child.prev_sibling().map(|c| c.kind()), child.kind()) {
            (_, SyntaxKind::SEMI) => buf.push_str(" { unimplemented!() }"),
            (_, SyntaxKind::ATTR) | (_, SyntaxKind::COMMENT) => {}
            (Some(SyntaxKind::ATTR), SyntaxKind::WHITESPACE)
            | (Some(SyntaxKind::COMMENT), SyntaxKind::WHITESPACE) => {}
            _ => child.text().push_to(&mut buf),
        };
    }

    buf.trim_end().to_string()
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
    fn foo(&self) { unimplemented!() }
    fn baz(&self) { unimplemented!() }<|>
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
    fn foo(&self) { unimplemented!() }<|>
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
    fn foo(&self) { unimplemented!() }<|>
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
    fn foo(&self) { unimplemented!() }<|>
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
    fn test_indented_impl_block() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo {
    fn valid(some: u32) -> bool;
}
struct S;

mod my_mod {
    impl crate::Foo for S { <|> }
}",
            "
trait Foo {
    fn valid(some: u32) -> bool;
}
struct S;

mod my_mod {
    impl crate::Foo for S {
        fn valid(some: u32) -> bool { unimplemented!() }<|>
    }
}",
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
    fn foo(&self) { unimplemented!() }<|>
}"#,
        )
    }
}
