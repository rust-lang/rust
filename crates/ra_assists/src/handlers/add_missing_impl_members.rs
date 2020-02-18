use hir::HasSource;
use ra_syntax::{
    ast::{self, edit, make, AstNode, NameOwner},
    SmolStr,
};

use crate::{
    ast_transform::{self, AstTransform, QualifyPaths, SubstituteTypeParams},
    utils::{get_missing_impl_items, resolve_target_trait},
    Assist, AssistCtx, AssistId,
};

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
pub(crate) fn add_missing_impl_members(ctx: AssistCtx) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::NoDefaultMethods,
        "add_impl_missing_members",
        "Implement missing members",
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
pub(crate) fn add_missing_default_members(ctx: AssistCtx) -> Option<Assist> {
    add_missing_impl_members_inner(
        ctx,
        AddMissingImplMembersMode::DefaultMethodsOnly,
        "add_impl_default_members",
        "Implement default members",
    )
}

fn add_missing_impl_members_inner(
    ctx: AssistCtx,
    mode: AddMissingImplMembersMode,
    assist_id: &'static str,
    label: &'static str,
) -> Option<Assist> {
    let _p = ra_prof::profile("add_missing_impl_members_inner");
    let impl_node = ctx.find_node_at_offset::<ast::ImplBlock>()?;
    let impl_item_list = impl_node.item_list()?;

    let trait_ = resolve_target_trait(&ctx.sema, &impl_node)?;

    let def_name = |item: &ast::ImplItem| -> Option<SmolStr> {
        match item {
            ast::ImplItem::FnDef(def) => def.name(),
            ast::ImplItem::TypeAliasDef(def) => def.name(),
            ast::ImplItem::ConstDef(def) => def.name(),
        }
        .map(|it| it.text().clone())
    };

    let missing_items = get_missing_impl_items(&ctx.sema, &impl_node)
        .iter()
        .map(|i| match i {
            hir::AssocItem::Function(i) => ast::ImplItem::FnDef(i.source(ctx.db).value),
            hir::AssocItem::TypeAlias(i) => ast::ImplItem::TypeAliasDef(i.source(ctx.db).value),
            hir::AssocItem::Const(i) => ast::ImplItem::ConstDef(i.source(ctx.db).value),
        })
        .filter(|t| def_name(&t).is_some())
        .filter(|t| match t {
            ast::ImplItem::FnDef(def) => match mode {
                AddMissingImplMembersMode::DefaultMethodsOnly => def.body().is_some(),
                AddMissingImplMembersMode::NoDefaultMethods => def.body().is_none(),
            },
            _ => mode == AddMissingImplMembersMode::NoDefaultMethods,
        })
        .collect::<Vec<_>>();

    if missing_items.is_empty() {
        return None;
    }

    let sema = ctx.sema;

    ctx.add_assist(AssistId(assist_id), label, |edit| {
        let n_existing_items = impl_item_list.impl_items().count();
        let source_scope = sema.scope_for_def(trait_);
        let target_scope = sema.scope(impl_item_list.syntax());
        let ast_transform = QualifyPaths::new(&target_scope, &source_scope, sema.db)
            .or(SubstituteTypeParams::for_trait_impl(&source_scope, sema.db, trait_, impl_node));
        let items = missing_items
            .into_iter()
            .map(|it| ast_transform::apply(&*ast_transform, it))
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

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_not_applicable};

    use super::*;

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
    fn test_qualify_path_1() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub struct Bar;
    trait Foo { fn foo(&self, bar: Bar); }
}
struct S;
impl foo::Foo for S { <|> }",
            "
mod foo {
    pub struct Bar;
    trait Foo { fn foo(&self, bar: Bar); }
}
struct S;
impl foo::Foo for S {
    <|>fn foo(&self, bar: foo::Bar) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_qualify_path_generic() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub struct Bar<T>;
    trait Foo { fn foo(&self, bar: Bar<u32>); }
}
struct S;
impl foo::Foo for S { <|> }",
            "
mod foo {
    pub struct Bar<T>;
    trait Foo { fn foo(&self, bar: Bar<u32>); }
}
struct S;
impl foo::Foo for S {
    <|>fn foo(&self, bar: foo::Bar<u32>) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_qualify_path_and_substitute_param() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub struct Bar<T>;
    trait Foo<T> { fn foo(&self, bar: Bar<T>); }
}
struct S;
impl foo::Foo<u32> for S { <|> }",
            "
mod foo {
    pub struct Bar<T>;
    trait Foo<T> { fn foo(&self, bar: Bar<T>); }
}
struct S;
impl foo::Foo<u32> for S {
    <|>fn foo(&self, bar: foo::Bar<u32>) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_substitute_param_no_qualify() {
        // when substituting params, the substituted param should not be qualified!
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    trait Foo<T> { fn foo(&self, bar: T); }
    pub struct Param;
}
struct Param;
struct S;
impl foo::Foo<Param> for S { <|> }",
            "
mod foo {
    trait Foo<T> { fn foo(&self, bar: T); }
    pub struct Param;
}
struct Param;
struct S;
impl foo::Foo<Param> for S {
    <|>fn foo(&self, bar: Param) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_qualify_path_associated_item() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub struct Bar<T>;
    impl Bar<T> { type Assoc = u32; }
    trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
}
struct S;
impl foo::Foo for S { <|> }",
            "
mod foo {
    pub struct Bar<T>;
    impl Bar<T> { type Assoc = u32; }
    trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
}
struct S;
impl foo::Foo for S {
    <|>fn foo(&self, bar: foo::Bar<u32>::Assoc) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_qualify_path_nested() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub struct Bar<T>;
    pub struct Baz;
    trait Foo { fn foo(&self, bar: Bar<Baz>); }
}
struct S;
impl foo::Foo for S { <|> }",
            "
mod foo {
    pub struct Bar<T>;
    pub struct Baz;
    trait Foo { fn foo(&self, bar: Bar<Baz>); }
}
struct S;
impl foo::Foo for S {
    <|>fn foo(&self, bar: foo::Bar<foo::Baz>) { unimplemented!() }
}",
        );
    }

    #[test]
    fn test_qualify_path_fn_trait_notation() {
        check_assist(
            add_missing_impl_members,
            "
mod foo {
    pub trait Fn<Args> { type Output; }
    trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
}
struct S;
impl foo::Foo for S { <|> }",
            "
mod foo {
    pub trait Fn<Args> { type Output; }
    trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
}
struct S;
impl foo::Foo for S {
    <|>fn foo(&self, bar: dyn Fn(u32) -> i32) { unimplemented!() }
}",
        );
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
