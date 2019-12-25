use std::collections::HashMap;

use hir::{db::HirDatabase, HasSource};
use ra_syntax::{
    ast::{self, edit, make, AstNode, NameOwner},
    SmolStr,
};

use crate::{Assist, AssistCtx, AssistId};

#[derive(PartialEq)]
enum AddMissingImplMembersMode {
    DefaultMethodsOnly,
    NoDefaultMethods,
}

// Assist: add_impl_missing_members
//
// Adds scaffold for required impl members.
//
// ```
// trait Trait<T> {
//     Type X;
//     fn foo(&self) -> T;
//     fn bar(&self) {}
// }
//
// impl Trait<u32> for () {<|>
//
// }
// ```
// ->
// ```
// trait Trait<T> {
//     Type X;
//     fn foo(&self) -> T;
//     fn bar(&self) {}
// }
//
// impl Trait<u32> for () {
//     fn foo(&self) -> u32 { unimplemented!() }
//
// }
// ```
pub(crate) fn add_missing_impl_members(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::NoDefaultMethods,
        "add_impl_missing_members",
        "add missing impl members",
    )
}

// Assist: add_impl_default_members
//
// Adds scaffold for overriding default impl members.
//
// ```
// trait Trait {
//     Type X;
//     fn foo(&self);
//     fn bar(&self) {}
// }
//
// impl Trait for () {
//     Type X = ();
//     fn foo(&self) {}<|>
//
// }
// ```
// ->
// ```
// trait Trait {
//     Type X;
//     fn foo(&self);
//     fn bar(&self) {}
// }
//
// impl Trait for () {
//     Type X = ();
//     fn foo(&self) {}
//     fn bar(&self) {}
//
// }
// ```
pub(crate) fn add_missing_default_members(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::DefaultMethodsOnly,
        "add_impl_default_members",
        "add impl default members",
    )
}

fn add_missing_impl_members_inner(
    ctx: AssistCtx<impl HirDatabase>,
    mode: AddMissingImplMembersMode,
    assist_id: &'static str,
    label: &'static str,
) -> Option<Assist> {
    let impl_node = ctx.find_node_at_offset::<ast::ImplBlock>()?;
    let impl_item_list = impl_node.item_list()?;

    let (trait_, trait_def) = {
        let analyzer = ctx.source_analyzer(impl_node.syntax(), None);

        resolve_target_trait_def(ctx.db, &analyzer, &impl_node)?
    };

    let def_name = |item: &ast::ImplItem| -> Option<SmolStr> {
        match item {
            ast::ImplItem::FnDef(def) => def.name(),
            ast::ImplItem::TypeAliasDef(def) => def.name(),
            ast::ImplItem::ConstDef(def) => def.name(),
        }
        .map(|it| it.text().clone())
    };

    let trait_items = trait_def.item_list()?.impl_items();
    let impl_items = impl_item_list.impl_items().collect::<Vec<_>>();

    let missing_items: Vec<_> = trait_items
        .filter(|t| def_name(t).is_some())
        .filter(|t| match t {
            ast::ImplItem::FnDef(def) => match mode {
                AddMissingImplMembersMode::DefaultMethodsOnly => def.body().is_some(),
                AddMissingImplMembersMode::NoDefaultMethods => def.body().is_none(),
            },
            _ => mode == AddMissingImplMembersMode::NoDefaultMethods,
        })
        .filter(|t| impl_items.iter().all(|i| def_name(i) != def_name(t)))
        .collect();
    if missing_items.is_empty() {
        return None;
    }

    let file_id = ctx.frange.file_id;
    let db = ctx.db;

    ctx.add_assist(AssistId(assist_id), label, |edit| {
        let n_existing_items = impl_item_list.impl_items().count();
        let substs = get_syntactic_substs(impl_node).unwrap_or_default();
        let generic_def: hir::GenericDef = trait_.into();
        let substs_by_param: HashMap<_, _> = generic_def
            .params(db)
            .into_iter()
            // this is a trait impl, so we need to skip the first type parameter -- this is a bit hacky
            .skip(1)
            .zip(substs.into_iter())
            .collect();
        let items = missing_items
            .into_iter()
            .map(|it| {
                substitute_type_params(db, hir::InFile::new(file_id.into(), it), &substs_by_param)
            })
            .map(|it| match it {
                ast::ImplItem::FnDef(def) => ast::ImplItem::FnDef(add_body(def)),
                _ => it,
            })
            .map(|it| edit::strip_attrs_and_docs(&it));
        let new_impl_item_list = impl_item_list.append_items(items);
        let cursor_position = {
            let first_new_item = new_impl_item_list.impl_items().nth(n_existing_items).unwrap();
            first_new_item.syntax().text_range().start()
        };

        edit.replace_ast(impl_item_list, new_impl_item_list);
        edit.set_cursor(cursor_position);
    })
}

fn add_body(fn_def: ast::FnDef) -> ast::FnDef {
    if fn_def.body().is_none() {
        fn_def.with_body(make::block_from_expr(make::expr_unimplemented()))
    } else {
        fn_def
    }
}

// FIXME: It would probably be nicer if we could get this via HIR (i.e. get the
// trait ref, and then go from the types in the substs back to the syntax)
// FIXME: This should be a general utility (not even just for assists)
fn get_syntactic_substs(impl_block: ast::ImplBlock) -> Option<Vec<ast::TypeRef>> {
    let target_trait = impl_block.target_trait()?;
    let path_type = match target_trait {
        ast::TypeRef::PathType(path) => path,
        _ => return None,
    };
    let type_arg_list = path_type.path()?.segment()?.type_arg_list()?;
    let mut result = Vec::new();
    for type_arg in type_arg_list.type_args() {
        let type_arg: ast::TypeArg = type_arg;
        result.push(type_arg.type_ref()?);
    }
    Some(result)
}

// FIXME: This should be a general utility (not even just for assists)
fn substitute_type_params<N: AstNode>(
    db: &impl HirDatabase,
    node: hir::InFile<N>,
    substs: &HashMap<hir::TypeParam, ast::TypeRef>,
) -> N {
    let type_param_replacements = node
        .value
        .syntax()
        .descendants()
        .filter_map(ast::TypeRef::cast)
        .filter_map(|n| {
            let path = match &n {
                ast::TypeRef::PathType(path_type) => path_type.path()?,
                _ => return None,
            };
            let analyzer = hir::SourceAnalyzer::new(db, node.with_value(n.syntax()), None);
            let resolution = analyzer.resolve_path(db, &path)?;
            match resolution {
                hir::PathResolution::TypeParam(tp) => Some((n, substs.get(&tp)?.clone())),
                _ => None,
            }
        })
        .collect::<Vec<_>>();

    if type_param_replacements.is_empty() {
        node.value
    } else {
        edit::replace_descendants(&node.value, type_param_replacements.into_iter())
    }
}

/// Given an `ast::ImplBlock`, resolves the target trait (the one being
/// implemented) to a `ast::TraitDef`.
fn resolve_target_trait_def(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    impl_block: &ast::ImplBlock,
) -> Option<(hir::Trait, ast::TraitDef)> {
    let ast_path = impl_block
        .target_trait()
        .map(|it| it.syntax().clone())
        .and_then(ast::PathType::cast)?
        .path()?;

    match analyzer.resolve_path(db, &ast_path) {
        Some(hir::PathResolution::Def(hir::ModuleDef::Trait(def))) => {
            Some((def, def.source(db).value))
        }
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
    type Output;

    const CONST: usize = 42;

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
    type Output;

    const CONST: usize = 42;

    fn foo(&self);
    fn bar(&self);
    fn baz(&self);
}

struct S;

impl Foo for S {
    fn bar(&self) {}
    <|>type Output;
    const CONST: usize = 42;
    fn foo(&self) { unimplemented!() }
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
    fn fill_in_type_params_1() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl Foo<u32> for S { <|> }",
            "
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl Foo<u32> for S {
    <|>fn foo(&self, t: u32) -> &u32 { unimplemented!() }
}",
        );
    }

    #[test]
    fn fill_in_type_params_2() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl<U> Foo<U> for S { <|> }",
            "
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl<U> Foo<U> for S {
    <|>fn foo(&self, t: U) -> &U { unimplemented!() }
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
    type Output;

    #[must_use]
    fn foo(&self);
}
struct S;
impl Foo for S {}<|>"#,
            r#"
#[doc(alias = "test alias")]
trait Foo {
    /// doc string
    type Output;

    #[must_use]
    fn foo(&self);
}
struct S;
impl Foo for S {
    <|>type Output;
    fn foo(&self) { unimplemented!() }
}"#,
        )
    }

    #[test]
    fn test_default_methods() {
        check_assist(
            add_missing_default_members,
            "
trait Foo {
    type Output;

    const CONST: usize = 42;

    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S { <|> }",
            "
trait Foo {
    type Output;

    const CONST: usize = 42;

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
