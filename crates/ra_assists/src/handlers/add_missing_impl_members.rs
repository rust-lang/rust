use hir::HasSource;
use ra_syntax::{
    ast::{
        self,
        edit::{self, AstNodeEdit, IndentLevel},
        make, AstNode, NameOwner,
    },
    SmolStr,
};

use crate::{
    assist_context::{AssistContext, Assists},
    ast_transform::{self, AstTransform, QualifyPaths, SubstituteTypeParams},
    utils::{get_missing_assoc_items, render_snippet, resolve_target_trait, Cursor},
    AssistId, AssistKind,
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
//     fn foo(&self) -> u32 {
//         ${0:todo!()}
//     }
//
// }
// ```
pub(crate) fn add_missing_impl_members(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    add_missing_impl_members_inner(
        acc,
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
//     $0fn bar(&self) {}
//
// }
// ```
pub(crate) fn add_missing_default_members(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    add_missing_impl_members_inner(
        acc,
        ctx,
        AddMissingImplMembersMode::DefaultMethodsOnly,
        "add_impl_default_members",
        "Implement default members",
    )
}

fn add_missing_impl_members_inner(
    acc: &mut Assists,
    ctx: &AssistContext,
    mode: AddMissingImplMembersMode,
    assist_id: &'static str,
    label: &'static str,
) -> Option<()> {
    let _p = ra_prof::profile("add_missing_impl_members_inner");
    let impl_def = ctx.find_node_at_offset::<ast::ImplDef>()?;
    let impl_item_list = impl_def.item_list()?;

    let trait_ = resolve_target_trait(&ctx.sema, &impl_def)?;

    let def_name = |item: &ast::AssocItem| -> Option<SmolStr> {
        match item {
            ast::AssocItem::FnDef(def) => def.name(),
            ast::AssocItem::TypeAliasDef(def) => def.name(),
            ast::AssocItem::ConstDef(def) => def.name(),
        }
        .map(|it| it.text().clone())
    };

    let missing_items = get_missing_assoc_items(&ctx.sema, &impl_def)
        .iter()
        .map(|i| match i {
            hir::AssocItem::Function(i) => ast::AssocItem::FnDef(i.source(ctx.db()).value),
            hir::AssocItem::TypeAlias(i) => ast::AssocItem::TypeAliasDef(i.source(ctx.db()).value),
            hir::AssocItem::Const(i) => ast::AssocItem::ConstDef(i.source(ctx.db()).value),
        })
        .filter(|t| def_name(&t).is_some())
        .filter(|t| match t {
            ast::AssocItem::FnDef(def) => match mode {
                AddMissingImplMembersMode::DefaultMethodsOnly => def.body().is_some(),
                AddMissingImplMembersMode::NoDefaultMethods => def.body().is_none(),
            },
            _ => mode == AddMissingImplMembersMode::NoDefaultMethods,
        })
        .collect::<Vec<_>>();

    if missing_items.is_empty() {
        return None;
    }

    let target = impl_def.syntax().text_range();
    acc.add(AssistId(assist_id, AssistKind::QuickFix), label, target, |builder| {
        let n_existing_items = impl_item_list.assoc_items().count();
        let source_scope = ctx.sema.scope_for_def(trait_);
        let target_scope = ctx.sema.scope(impl_item_list.syntax());
        let ast_transform = QualifyPaths::new(&target_scope, &source_scope)
            .or(SubstituteTypeParams::for_trait_impl(&source_scope, trait_, impl_def));
        let items = missing_items
            .into_iter()
            .map(|it| ast_transform::apply(&*ast_transform, it))
            .map(|it| match it {
                ast::AssocItem::FnDef(def) => ast::AssocItem::FnDef(add_body(def)),
                _ => it,
            })
            .map(|it| edit::remove_attrs_and_docs(&it));
        let new_impl_item_list = impl_item_list.append_items(items);
        let first_new_item = new_impl_item_list.assoc_items().nth(n_existing_items).unwrap();

        let original_range = impl_item_list.syntax().text_range();
        match ctx.config.snippet_cap {
            None => builder.replace(original_range, new_impl_item_list.to_string()),
            Some(cap) => {
                let mut cursor = Cursor::Before(first_new_item.syntax());
                let placeholder;
                if let ast::AssocItem::FnDef(func) = &first_new_item {
                    if let Some(m) = func.syntax().descendants().find_map(ast::MacroCall::cast) {
                        if m.syntax().text() == "todo!()" {
                            placeholder = m;
                            cursor = Cursor::Replace(placeholder.syntax());
                        }
                    }
                }
                builder.replace_snippet(
                    cap,
                    original_range,
                    render_snippet(cap, new_impl_item_list.syntax(), cursor),
                )
            }
        };
    })
}

fn add_body(fn_def: ast::FnDef) -> ast::FnDef {
    if fn_def.body().is_some() {
        return fn_def;
    }
    let body = make::block_expr(None, Some(make::expr_todo())).indent(IndentLevel(1));
    fn_def.with_body(body)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_add_missing_impl_members() {
        check_assist(
            add_missing_impl_members,
            r#"
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
}"#,
            r#"
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
    $0type Output;
    const CONST: usize = 42;
    fn foo(&self) {
        todo!()
    }
    fn baz(&self) {
        todo!()
    }

}"#,
        );
    }

    #[test]
    fn test_copied_overriden_members() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo {
    fn foo(&self);
    fn bar(&self) -> bool { true }
    fn baz(&self) -> u32 { 42 }
}

struct S;

impl Foo for S {
    fn bar(&self) {}
<|>
}"#,
            r#"
trait Foo {
    fn foo(&self);
    fn bar(&self) -> bool { true }
    fn baz(&self) -> u32 { 42 }
}

struct S;

impl Foo for S {
    fn bar(&self) {}
    fn foo(&self) {
        ${0:todo!()}
    }

}"#,
        );
    }

    #[test]
    fn test_empty_impl_def() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo { fn foo(&self); }
struct S;
impl Foo for S { <|> }"#,
            r#"
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {
    fn foo(&self) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn fill_in_type_params_1() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl Foo<u32> for S { <|> }"#,
            r#"
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl Foo<u32> for S {
    fn foo(&self, t: u32) -> &u32 {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn fill_in_type_params_2() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl<U> Foo<U> for S { <|> }"#,
            r#"
trait Foo<T> { fn foo(&self, t: T) -> &T; }
struct S;
impl<U> Foo<U> for S {
    fn foo(&self, t: U) -> &U {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_cursor_after_empty_impl_def() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {}<|>"#,
            r#"
trait Foo { fn foo(&self); }
struct S;
impl Foo for S {
    fn foo(&self) {
        ${0:todo!()}
    }
}"#,
        )
    }

    #[test]
    fn test_qualify_path_1() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub struct Bar;
    trait Foo { fn foo(&self, bar: Bar); }
}
struct S;
impl foo::Foo for S { <|> }"#,
            r#"
mod foo {
    pub struct Bar;
    trait Foo { fn foo(&self, bar: Bar); }
}
struct S;
impl foo::Foo for S {
    fn foo(&self, bar: foo::Bar) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_path_generic() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub struct Bar<T>;
    trait Foo { fn foo(&self, bar: Bar<u32>); }
}
struct S;
impl foo::Foo for S { <|> }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    trait Foo { fn foo(&self, bar: Bar<u32>); }
}
struct S;
impl foo::Foo for S {
    fn foo(&self, bar: foo::Bar<u32>) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_path_and_substitute_param() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub struct Bar<T>;
    trait Foo<T> { fn foo(&self, bar: Bar<T>); }
}
struct S;
impl foo::Foo<u32> for S { <|> }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    trait Foo<T> { fn foo(&self, bar: Bar<T>); }
}
struct S;
impl foo::Foo<u32> for S {
    fn foo(&self, bar: foo::Bar<u32>) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_substitute_param_no_qualify() {
        // when substituting params, the substituted param should not be qualified!
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    trait Foo<T> { fn foo(&self, bar: T); }
    pub struct Param;
}
struct Param;
struct S;
impl foo::Foo<Param> for S { <|> }"#,
            r#"
mod foo {
    trait Foo<T> { fn foo(&self, bar: T); }
    pub struct Param;
}
struct Param;
struct S;
impl foo::Foo<Param> for S {
    fn foo(&self, bar: Param) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_path_associated_item() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub struct Bar<T>;
    impl Bar<T> { type Assoc = u32; }
    trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
}
struct S;
impl foo::Foo for S { <|> }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    impl Bar<T> { type Assoc = u32; }
    trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
}
struct S;
impl foo::Foo for S {
    fn foo(&self, bar: foo::Bar<u32>::Assoc) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_path_nested() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub struct Bar<T>;
    pub struct Baz;
    trait Foo { fn foo(&self, bar: Bar<Baz>); }
}
struct S;
impl foo::Foo for S { <|> }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    pub struct Baz;
    trait Foo { fn foo(&self, bar: Bar<Baz>); }
}
struct S;
impl foo::Foo for S {
    fn foo(&self, bar: foo::Bar<foo::Baz>) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_path_fn_trait_notation() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub trait Fn<Args> { type Output; }
    trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
}
struct S;
impl foo::Foo for S { <|> }"#,
            r#"
mod foo {
    pub trait Fn<Args> { type Output; }
    trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
}
struct S;
impl foo::Foo for S {
    fn foo(&self, bar: dyn Fn(u32) -> i32) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_empty_trait() {
        check_assist_not_applicable(
            add_missing_impl_members,
            r#"
trait Foo;
struct S;
impl Foo for S { <|> }"#,
        )
    }

    #[test]
    fn test_ignore_unnamed_trait_members_and_default_methods() {
        check_assist_not_applicable(
            add_missing_impl_members,
            r#"
trait Foo {
    fn (arg: u32);
    fn valid(some: u32) -> bool { false }
}
struct S;
impl Foo for S { <|> }"#,
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
    $0type Output;
    fn foo(&self) {
        todo!()
    }
}"#,
        )
    }

    #[test]
    fn test_default_methods() {
        check_assist(
            add_missing_default_members,
            r#"
trait Foo {
    type Output;

    const CONST: usize = 42;

    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S { <|> }"#,
            r#"
trait Foo {
    type Output;

    const CONST: usize = 42;

    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S {
    $0fn valid(some: u32) -> bool { false }
}"#,
        )
    }

    #[test]
    fn test_generic_single_default_parameter() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<T = Self> {
    fn bar(&self, other: &T);
}

struct S;
impl Foo for S { <|> }"#,
            r#"
trait Foo<T = Self> {
    fn bar(&self, other: &T);
}

struct S;
impl Foo for S {
    fn bar(&self, other: &Self) {
        ${0:todo!()}
    }
}"#,
        )
    }

    #[test]
    fn test_generic_default_parameter_is_second() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<T1, T2 = Self> {
    fn bar(&self, this: &T1, that: &T2);
}

struct S<T>;
impl Foo<T> for S<T> { <|> }"#,
            r#"
trait Foo<T1, T2 = Self> {
    fn bar(&self, this: &T1, that: &T2);
}

struct S<T>;
impl Foo<T> for S<T> {
    fn bar(&self, this: &T, that: &Self) {
        ${0:todo!()}
    }
}"#,
        )
    }
}
