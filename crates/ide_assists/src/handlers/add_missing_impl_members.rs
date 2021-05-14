use ide_db::traits::resolve_target_trait;
use syntax::ast::{self, AstNode};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::{
        add_trait_assoc_items_to_impl, filter_assoc_items, render_snippet, Cursor, DefaultMethods,
    },
    AssistId, AssistKind,
};

// Assist: add_impl_missing_members
//
// Adds scaffold for required impl members.
//
// ```
// trait Trait<T> {
//     type X;
//     fn foo(&self) -> T;
//     fn bar(&self) {}
// }
//
// impl Trait<u32> for () {$0
//
// }
// ```
// ->
// ```
// trait Trait<T> {
//     type X;
//     fn foo(&self) -> T;
//     fn bar(&self) {}
// }
//
// impl Trait<u32> for () {
//     $0type X;
//
//     fn foo(&self) -> u32 {
//         todo!()
//     }
// }
// ```
pub(crate) fn add_missing_impl_members(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    add_missing_impl_members_inner(
        acc,
        ctx,
        DefaultMethods::No,
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
//     type X;
//     fn foo(&self);
//     fn bar(&self) {}
// }
//
// impl Trait for () {
//     type X = ();
//     fn foo(&self) {}$0
// }
// ```
// ->
// ```
// trait Trait {
//     type X;
//     fn foo(&self);
//     fn bar(&self) {}
// }
//
// impl Trait for () {
//     type X = ();
//     fn foo(&self) {}
//
//     $0fn bar(&self) {}
// }
// ```
pub(crate) fn add_missing_default_members(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    add_missing_impl_members_inner(
        acc,
        ctx,
        DefaultMethods::Only,
        "add_impl_default_members",
        "Implement default members",
    )
}

fn add_missing_impl_members_inner(
    acc: &mut Assists,
    ctx: &AssistContext,
    mode: DefaultMethods,
    assist_id: &'static str,
    label: &'static str,
) -> Option<()> {
    let _p = profile::span("add_missing_impl_members_inner");
    let impl_def = ctx.find_node_at_offset::<ast::Impl>()?;
    let trait_ = resolve_target_trait(&ctx.sema, &impl_def)?;

    let missing_items = filter_assoc_items(
        ctx.db(),
        &ide_db::traits::get_missing_assoc_items(&ctx.sema, &impl_def),
        mode,
    );

    if missing_items.is_empty() {
        return None;
    }

    let target = impl_def.syntax().text_range();
    acc.add(AssistId(assist_id, AssistKind::QuickFix), label, target, |builder| {
        let target_scope = ctx.sema.scope(impl_def.syntax());
        let (new_impl_def, first_new_item) =
            add_trait_assoc_items_to_impl(&ctx.sema, missing_items, trait_, impl_def, target_scope);
        match ctx.config.snippet_cap {
            None => builder.replace(target, new_impl_def.to_string()),
            Some(cap) => {
                let mut cursor = Cursor::Before(first_new_item.syntax());
                let placeholder;
                if let ast::AssocItem::Fn(func) = &first_new_item {
                    if let Some(m) = func.syntax().descendants().find_map(ast::MacroCall::cast) {
                        if m.syntax().text() == "todo!()" {
                            placeholder = m;
                            cursor = Cursor::Replace(placeholder.syntax());
                        }
                    }
                }
                builder.replace_snippet(
                    cap,
                    target,
                    render_snippet(cap, new_impl_def.syntax(), cursor),
                )
            }
        };
    })
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
$0
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
$0
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
impl Foo for S { $0 }"#,
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
    fn test_impl_def_without_braces() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo { fn foo(&self); }
struct S;
impl Foo for S$0"#,
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
impl Foo<u32> for S { $0 }"#,
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
impl<U> Foo<U> for S { $0 }"#,
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
impl Foo for S {}$0"#,
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
impl foo::Foo for S { $0 }"#,
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
    fn test_qualify_path_2() {
        check_assist(
            add_missing_impl_members,
            r#"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub trait Foo { fn foo(&self, bar: Bar); }
    }
}

use foo::bar;

struct S;
impl bar::Foo for S { $0 }"#,
            r#"
mod foo {
    pub mod bar {
        pub struct Bar;
        pub trait Foo { fn foo(&self, bar: Bar); }
    }
}

use foo::bar;

struct S;
impl bar::Foo for S {
    fn foo(&self, bar: bar::Bar) {
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
impl foo::Foo for S { $0 }"#,
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
impl foo::Foo<u32> for S { $0 }"#,
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
impl foo::Foo<Param> for S { $0 }"#,
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
impl foo::Foo for S { $0 }"#,
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
impl foo::Foo for S { $0 }"#,
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
impl foo::Foo for S { $0 }"#,
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
impl Foo for S { $0 }"#,
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
impl Foo for S { $0 }"#,
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
impl Foo for S {}$0"#,
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
impl Foo for S { $0 }"#,
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
impl Foo for S { $0 }"#,
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
impl Foo<T> for S<T> { $0 }"#,
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

    #[test]
    fn test_assoc_type_bounds_are_removed() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    type Ty: Copy + 'static;
}

impl Tr for ()$0 {
}"#,
            r#"
trait Tr {
    type Ty: Copy + 'static;
}

impl Tr for () {
    $0type Ty;
}"#,
        )
    }

    #[test]
    fn test_whitespace_fixup_preserves_bad_tokens() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    fn foo();
}

impl Tr for ()$0 {
    +++
}"#,
            r#"
trait Tr {
    fn foo();
}

impl Tr for () {
    fn foo() {
        ${0:todo!()}
    }
    +++
}"#,
        )
    }

    #[test]
    fn test_whitespace_fixup_preserves_comments() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    fn foo();
}

impl Tr for ()$0 {
    // very important
}"#,
            r#"
trait Tr {
    fn foo();
}

impl Tr for () {
    fn foo() {
        ${0:todo!()}
    }
    // very important
}"#,
        )
    }

    #[test]
    fn weird_path() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Test {
    fn foo(&self, x: crate)
}
impl Test for () {
    $0
}
"#,
            r#"
trait Test {
    fn foo(&self, x: crate)
}
impl Test for () {
    fn foo(&self, x: crate) {
        ${0:todo!()}
    }
}
"#,
        )
    }

    #[test]
    fn missing_generic_type() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<BAR> {
    fn foo(&self, bar: BAR);
}
impl Foo for () {
    $0
}
"#,
            r#"
trait Foo<BAR> {
    fn foo(&self, bar: BAR);
}
impl Foo for () {
    fn foo(&self, bar: BAR) {
        ${0:todo!()}
    }
}
"#,
        )
    }
}
