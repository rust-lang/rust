use hir::HasSource;
use syntax::ast::{self, make, AstNode};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::{add_trait_assoc_items_to_impl, filter_assoc_items, gen_trait_fn_body, DefaultMethods},
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
pub(crate) fn add_missing_impl_members(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
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
pub(crate) fn add_missing_default_members(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
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
    ctx: &AssistContext<'_>,
    mode: DefaultMethods,
    assist_id: &'static str,
    label: &'static str,
) -> Option<()> {
    let _p = profile::span("add_missing_impl_members_inner");
    let impl_def = ctx.find_node_at_offset::<ast::Impl>()?;
    let impl_ = ctx.sema.to_def(&impl_def)?;

    if ctx.token_at_offset().all(|t| {
        t.parent_ancestors()
            .take_while(|node| node != impl_def.syntax())
            .any(|s| ast::BlockExpr::can_cast(s.kind()) || ast::ParamList::can_cast(s.kind()))
    }) {
        return None;
    }

    let target_scope = ctx.sema.scope(impl_def.syntax())?;
    let trait_ref = impl_.trait_ref(ctx.db())?;
    let trait_ = trait_ref.trait_();

    let missing_items = filter_assoc_items(
        &ctx.sema,
        &ide_db::traits::get_missing_assoc_items(&ctx.sema, &impl_def),
        mode,
    );

    if missing_items.is_empty() {
        return None;
    }

    let target = impl_def.syntax().text_range();
    acc.add(AssistId(assist_id, AssistKind::QuickFix), label, target, |edit| {
        let new_impl_def = edit.make_mut(impl_def.clone());
        let first_new_item = add_trait_assoc_items_to_impl(
            &ctx.sema,
            &missing_items,
            trait_,
            &new_impl_def,
            target_scope,
        );

        if let Some(cap) = ctx.config.snippet_cap {
            let mut placeholder = None;
            if let DefaultMethods::No = mode {
                if let ast::AssocItem::Fn(func) = &first_new_item {
                    if try_gen_trait_body(ctx, func, trait_ref, &impl_def).is_none() {
                        if let Some(m) = func.syntax().descendants().find_map(ast::MacroCall::cast)
                        {
                            if m.syntax().text() == "todo!()" {
                                placeholder = Some(m);
                            }
                        }
                    }
                }
            }

            if let Some(macro_call) = placeholder {
                edit.add_placeholder_snippet(cap, macro_call);
            } else {
                edit.add_tabstop_before(cap, first_new_item);
            };
        };
    })
}

fn try_gen_trait_body(
    ctx: &AssistContext<'_>,
    func: &ast::Fn,
    trait_ref: hir::TraitRef,
    impl_def: &ast::Impl,
) -> Option<()> {
    let trait_path =
        make::ext::ident_path(&trait_ref.trait_().name(ctx.db()).display(ctx.db()).to_string());
    let hir_ty = ctx.sema.resolve_type(&impl_def.self_ty()?)?;
    let adt = hir_ty.as_adt()?.source(ctx.db())?;
    gen_trait_fn_body(func, &trait_path, &adt.value, Some(trait_ref))
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
    const CONST_2: i32;

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
    const CONST_2: i32;

    fn foo(&self);
    fn bar(&self);
    fn baz(&self);
}

struct S;

impl Foo for S {
    fn bar(&self) {}

    $0type Output;

    const CONST_2: i32;

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
    fn test_copied_overridden_members() {
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
    fn test_lifetime_substitution() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Trait<'a, 'b, A, B, C> {
    fn foo(&self, one: &'a A, anoter: &'b B) -> &'a C;
}

impl<'x, 'y, T, V, U> Trait<'x, 'y, T, V, U> for () {$0}"#,
            r#"
pub trait Trait<'a, 'b, A, B, C> {
    fn foo(&self, one: &'a A, anoter: &'b B) -> &'a C;
}

impl<'x, 'y, T, V, U> Trait<'x, 'y, T, V, U> for () {
    fn foo(&self, one: &'x T, anoter: &'y V) -> &'x U {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_lifetime_substitution_with_body() {
        check_assist(
            add_missing_default_members,
            r#"
pub trait Trait<'a, 'b, A, B, C: Default> {
    fn foo(&self, _one: &'a A, _anoter: &'b B) -> (C, &'a i32) {
        let value: &'a i32 = &0;
        (C::default(), value)
    }
}

impl<'x, 'y, T, V, U: Default> Trait<'x, 'y, T, V, U> for () {$0}"#,
            r#"
pub trait Trait<'a, 'b, A, B, C: Default> {
    fn foo(&self, _one: &'a A, _anoter: &'b B) -> (C, &'a i32) {
        let value: &'a i32 = &0;
        (C::default(), value)
    }
}

impl<'x, 'y, T, V, U: Default> Trait<'x, 'y, T, V, U> for () {
    $0fn foo(&self, _one: &'x T, _anoter: &'y V) -> (U, &'x i32) {
        let value: &'x i32 = &0;
        (<U>::default(), value)
    }
}"#,
        );
    }

    #[test]
    fn test_const_substitution() {
        check_assist(
            add_missing_default_members,
            r#"
struct Bar<const: N: bool> {
    bar: [i32, N]
}

trait Foo<const N: usize, T> {
    fn get_n_sq(&self, arg: &T) -> usize { N * N }
    fn get_array(&self, arg: Bar<N>) -> [i32; N] { [1; N] }
}

struct S<T> {
    wrapped: T
}

impl<const X: usize, Y, Z> Foo<X, Z> for S<Y> {
    $0
}"#,
            r#"
struct Bar<const: N: bool> {
    bar: [i32, N]
}

trait Foo<const N: usize, T> {
    fn get_n_sq(&self, arg: &T) -> usize { N * N }
    fn get_array(&self, arg: Bar<N>) -> [i32; N] { [1; N] }
}

struct S<T> {
    wrapped: T
}

impl<const X: usize, Y, Z> Foo<X, Z> for S<Y> {
    $0fn get_n_sq(&self, arg: &Z) -> usize { X * X }

    fn get_array(&self, arg: Bar<X>) -> [i32; X] { [1; X] }
}"#,
        )
    }

    #[test]
    fn test_const_substitution_2() {
        check_assist(
            add_missing_default_members,
            r#"
trait Foo<const N: usize, const M: usize, T> {
    fn get_sum(&self, arg: &T) -> usize { N + M }
}

impl<X> Foo<42, {20 + 22}, X> for () {
    $0
}"#,
            r#"
trait Foo<const N: usize, const M: usize, T> {
    fn get_sum(&self, arg: &T) -> usize { N + M }
}

impl<X> Foo<42, {20 + 22}, X> for () {
    $0fn get_sum(&self, arg: &X) -> usize { 42 + {20 + 22} }
}"#,
        )
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
    pub trait Foo { fn foo(&self, bar: Bar); }
}
struct S;
impl foo::Foo for S { $0 }"#,
            r#"
mod foo {
    pub struct Bar;
    pub trait Foo { fn foo(&self, bar: Bar); }
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
    pub trait Foo { fn foo(&self, bar: Bar<u32>); }
}
struct S;
impl foo::Foo for S { $0 }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    pub trait Foo { fn foo(&self, bar: Bar<u32>); }
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
    pub trait Foo<T> { fn foo(&self, bar: Bar<T>); }
}
struct S;
impl foo::Foo<u32> for S { $0 }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    pub trait Foo<T> { fn foo(&self, bar: Bar<T>); }
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
    pub trait Foo<T> { fn foo(&self, bar: T); }
    pub struct Param;
}
struct Param;
struct S;
impl foo::Foo<Param> for S { $0 }"#,
            r#"
mod foo {
    pub trait Foo<T> { fn foo(&self, bar: T); }
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
    pub trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
}
struct S;
impl foo::Foo for S { $0 }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    impl Bar<T> { type Assoc = u32; }
    pub trait Foo { fn foo(&self, bar: Bar<u32>::Assoc); }
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
    pub trait Foo { fn foo(&self, bar: Bar<Baz>); }
}
struct S;
impl foo::Foo for S { $0 }"#,
            r#"
mod foo {
    pub struct Bar<T>;
    pub struct Baz;
    pub trait Foo { fn foo(&self, bar: Bar<Baz>); }
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
    pub trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
}
struct S;
impl foo::Foo for S { $0 }"#,
            r#"
mod foo {
    pub trait Fn<Args> { type Output; }
    pub trait Foo { fn foo(&self, bar: dyn Fn(u32) -> i32); }
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
    const CONST_2: i32;

    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S { $0 }"#,
            r#"
trait Foo {
    type Output;

    const CONST: usize = 42;
    const CONST_2: i32;

    fn valid(some: u32) -> bool { false }
    fn foo(some: u32) -> bool;
}
struct S;
impl Foo for S {
    $0const CONST: usize = 42;

    fn valid(some: u32) -> bool { false }
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
    fn test_qualify_generic_default_parameter() {
        check_assist(
            add_missing_impl_members,
            r#"
mod m {
    pub struct S;
    pub trait Foo<T = S> {
        fn bar(&self, other: &T);
    }
}

struct S;
impl m::Foo for S { $0 }"#,
            r#"
mod m {
    pub struct S;
    pub trait Foo<T = S> {
        fn bar(&self, other: &T);
    }
}

struct S;
impl m::Foo for S {
    fn bar(&self, other: &m::S) {
        ${0:todo!()}
    }
}"#,
        )
    }

    #[test]
    fn test_qualify_generic_default_parameter_2() {
        check_assist(
            add_missing_impl_members,
            r#"
mod m {
    pub struct Wrapper<T, V> {
        one: T,
        another: V
    };
    pub struct S;
    pub trait Foo<T = Wrapper<S, bool>> {
        fn bar(&self, other: &T);
    }
}

struct S;
impl m::Foo for S { $0 }"#,
            r#"
mod m {
    pub struct Wrapper<T, V> {
        one: T,
        another: V
    };
    pub struct S;
    pub trait Foo<T = Wrapper<S, bool>> {
        fn bar(&self, other: &T);
    }
}

struct S;
impl m::Foo for S {
    fn bar(&self, other: &m::Wrapper<m::S, bool>) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_qualify_generic_default_parameter_3() {
        check_assist(
            add_missing_impl_members,
            r#"
mod m {
    pub struct Wrapper<T, V> {
        one: T,
        another: V
    };
    pub struct S;
    pub trait Foo<T = S, V = Wrapper<T, S>> {
        fn bar(&self, other: &V);
    }
}

struct S;
impl m::Foo for S { $0 }"#,
            r#"
mod m {
    pub struct Wrapper<T, V> {
        one: T,
        another: V
    };
    pub struct S;
    pub trait Foo<T = S, V = Wrapper<T, S>> {
        fn bar(&self, other: &V);
    }
}

struct S;
impl m::Foo for S {
    fn bar(&self, other: &m::Wrapper<m::S, m::S>) {
        ${0:todo!()}
    }
}"#,
        );
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

    #[test]
    fn does_not_requalify_self_as_crate() {
        check_assist(
            add_missing_default_members,
            r"
struct Wrapper<T>(T);

trait T {
    fn f(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for () {
    $0
}
",
            r"
struct Wrapper<T>(T);

trait T {
    fn f(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for () {
    $0fn f(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}
",
        );
    }

    #[test]
    fn test_default_body_generation() {
        check_assist(
            add_missing_impl_members,
            r#"
//- minicore: default
struct Foo(usize);

impl Default for Foo {
    $0
}
"#,
            r#"
struct Foo(usize);

impl Default for Foo {
    $0fn default() -> Self {
        Self(Default::default())
    }
}
"#,
        )
    }

    #[test]
    fn test_from_macro() {
        check_assist(
            add_missing_default_members,
            r#"
macro_rules! foo {
    () => {
        trait FooB {
            fn foo<'lt>(&'lt self) {}
        }
    }
}
foo!();
struct Foo(usize);

impl FooB for Foo {
    $0
}
"#,
            r#"
macro_rules! foo {
    () => {
        trait FooB {
            fn foo<'lt>(&'lt self) {}
        }
    }
}
foo!();
struct Foo(usize);

impl FooB for Foo {
    $0fn foo<'lt>(&'lt self){}
}
"#,
        )
    }

    #[test]
    fn test_assoc_type_when_trait_with_same_name_in_scope() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Foo {}

pub trait Types {
    type Foo;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl<T: Types> Behavior<T> for Impl { $0 }"#,
            r#"
pub trait Foo {}

pub trait Types {
    type Foo;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl<T: Types> Behavior<T> for Impl {
    fn reproduce(&self, foo: <T as Types>::Foo) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_type_on_concrete_type() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Types {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl Behavior<u32> for Impl { $0 }"#,
            r#"
pub trait Types {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl Behavior<u32> for Impl {
    fn reproduce(&self, foo: <u32 as Types>::Foo) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_type_on_concrete_type_qualified() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Types {
    type Foo;
}

impl Types for std::string::String {
    type Foo = bool;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl Behavior<std::string::String> for Impl { $0 }"#,
            r#"
pub trait Types {
    type Foo;
}

impl Types for std::string::String {
    type Foo = bool;
}

pub trait Behavior<T: Types> {
    fn reproduce(&self, foo: T::Foo);
}

pub struct Impl;

impl Behavior<std::string::String> for Impl {
    fn reproduce(&self, foo: <std::string::String as Types>::Foo) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_type_on_concrete_type_multi_option_ambiguous() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Types {
    type Foo;
}

pub trait Types2 {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

impl Types2 for u32 {
    type Foo = String;
}

pub trait Behavior<T: Types + Types2> {
    fn reproduce(&self, foo: <T as Types2>::Foo);
}

pub struct Impl;

impl Behavior<u32> for Impl { $0 }"#,
            r#"
pub trait Types {
    type Foo;
}

pub trait Types2 {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

impl Types2 for u32 {
    type Foo = String;
}

pub trait Behavior<T: Types + Types2> {
    fn reproduce(&self, foo: <T as Types2>::Foo);
}

pub struct Impl;

impl Behavior<u32> for Impl {
    fn reproduce(&self, foo: <u32 as Types2>::Foo) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_type_on_concrete_type_multi_option() {
        check_assist(
            add_missing_impl_members,
            r#"
pub trait Types {
    type Foo;
}

pub trait Types2 {
    type Bar;
}

impl Types for u32 {
    type Foo = bool;
}

impl Types2 for u32 {
    type Bar = String;
}

pub trait Behavior<T: Types + Types2> {
    fn reproduce(&self, foo: T::Bar);
}

pub struct Impl;

impl Behavior<u32> for Impl { $0 }"#,
            r#"
pub trait Types {
    type Foo;
}

pub trait Types2 {
    type Bar;
}

impl Types for u32 {
    type Foo = bool;
}

impl Types2 for u32 {
    type Bar = String;
}

pub trait Behavior<T: Types + Types2> {
    fn reproduce(&self, foo: T::Bar);
}

pub struct Impl;

impl Behavior<u32> for Impl {
    fn reproduce(&self, foo: <u32 as Types2>::Bar) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_assoc_type_on_concrete_type_multi_option_foreign() {
        check_assist(
            add_missing_impl_members,
            r#"
mod bar {
    pub trait Types2 {
        type Bar;
    }
}

pub trait Types {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

impl bar::Types2 for u32 {
    type Bar = String;
}

pub trait Behavior<T: Types + bar::Types2> {
    fn reproduce(&self, foo: T::Bar);
}

pub struct Impl;

impl Behavior<u32> for Impl { $0 }"#,
            r#"
mod bar {
    pub trait Types2 {
        type Bar;
    }
}

pub trait Types {
    type Foo;
}

impl Types for u32 {
    type Foo = bool;
}

impl bar::Types2 for u32 {
    type Bar = String;
}

pub trait Behavior<T: Types + bar::Types2> {
    fn reproduce(&self, foo: T::Bar);
}

pub struct Impl;

impl Behavior<u32> for Impl {
    fn reproduce(&self, foo: <u32 as bar::Types2>::Bar) {
        ${0:todo!()}
    }
}"#,
        );
    }

    #[test]
    fn test_transform_path_in_path_expr() {
        check_assist(
            add_missing_default_members,
            r#"
pub trait Const {
    const FOO: u32;
}

pub trait Trait<T: Const> {
    fn foo() -> bool {
        match T::FOO {
            0 => true,
            _ => false,
        }
    }
}

impl Const for u32 {
    const FOO: u32 = 1;
}

struct Impl;

impl Trait<u32> for Impl { $0 }"#,
            r#"
pub trait Const {
    const FOO: u32;
}

pub trait Trait<T: Const> {
    fn foo() -> bool {
        match T::FOO {
            0 => true,
            _ => false,
        }
    }
}

impl Const for u32 {
    const FOO: u32 = 1;
}

struct Impl;

impl Trait<u32> for Impl {
    $0fn foo() -> bool {
        match <u32 as Const>::FOO {
            0 => true,
            _ => false,
        }
    }
}"#,
        );
    }

    #[test]
    fn test_default_partial_eq() {
        check_assist(
            add_missing_default_members,
            r#"
//- minicore: eq
struct SomeStruct {
    data: usize,
    field: (usize, usize),
}
impl PartialEq for SomeStruct {$0}
"#,
            r#"
struct SomeStruct {
    data: usize,
    field: (usize, usize),
}
impl PartialEq for SomeStruct {
    $0fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}
"#,
        );
    }

    #[test]
    fn test_partial_eq_body_when_types_semantically_match() {
        check_assist(
            add_missing_impl_members,
            r#"
//- minicore: eq
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, T> {$0}
"#,
            r#"
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, T> {
    $0fn eq(&self, other: &Alias<T>) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
"#,
        );
    }

    #[test]
    fn test_partial_eq_body_when_types_dont_match() {
        check_assist(
            add_missing_impl_members,
            r#"
//- minicore: eq
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, i32> {$0}
"#,
            r#"
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, i32> {
    fn eq(&self, other: &Alias<T>) -> bool {
        ${0:todo!()}
    }
}
"#,
        );
    }

    #[test]
    fn test_ignore_function_body() {
        check_assist_not_applicable(
            add_missing_default_members,
            r#"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    type X = u8;
    fn foo(&self) {$0
        let x = 5;
    }
}"#,
        )
    }

    #[test]
    fn test_ignore_param_list() {
        check_assist_not_applicable(
            add_missing_impl_members,
            r#"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self);
}

impl Trait for () {
    type X = u8;
    fn foo(&self$0) {
        let x = 5;
    }
}"#,
        )
    }

    #[test]
    fn test_ignore_scope_inside_function() {
        check_assist_not_applicable(
            add_missing_impl_members,
            r#"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self);
}

impl Trait for () {
    type X = u8;
    fn foo(&self) {
        let x = async {$0 5 };
    }
}"#,
        )
    }

    #[test]
    fn test_apply_outside_function() {
        check_assist(
            add_missing_default_members,
            r#"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    type X = u8;
    fn foo(&self)$0 {}
}"#,
            r#"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    type X = u8;
    fn foo(&self) {}

    $0fn bar(&self) {}
}"#,
        )
    }

    #[test]
    fn test_works_inside_function() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        $0
    }
}
"#,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        fn method() {
            ${0:todo!()}
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_add_missing_preserves_indentation() {
        // in different modules
        check_assist(
            add_missing_impl_members,
            r#"
mod m {
    pub trait Foo {
        const CONST_MULTILINE: (
            i32,
            i32
        );

        fn foo(&self);
    }
}
struct S;
impl m::Foo for S { $0 }"#,
            r#"
mod m {
    pub trait Foo {
        const CONST_MULTILINE: (
            i32,
            i32
        );

        fn foo(&self);
    }
}
struct S;
impl m::Foo for S {
    $0const CONST_MULTILINE: (
        i32,
        i32
    );

    fn foo(&self) {
        todo!()
    }
}"#,
        );
        // in the same module
        check_assist(
            add_missing_impl_members,
            r#"
mod m {
    trait Foo {
        type Output;

        const CONST: usize = 42;
        const CONST_2: i32;
        const CONST_MULTILINE: (
            i32,
            i32
        );

        fn foo(&self);
        fn bar(&self);
        fn baz(&self);
    }

    struct S;

    impl Foo for S {
        fn bar(&self) {}
$0
    }
}"#,
            r#"
mod m {
    trait Foo {
        type Output;

        const CONST: usize = 42;
        const CONST_2: i32;
        const CONST_MULTILINE: (
            i32,
            i32
        );

        fn foo(&self);
        fn bar(&self);
        fn baz(&self);
    }

    struct S;

    impl Foo for S {
        fn bar(&self) {}

        $0type Output;

        const CONST_2: i32;

        const CONST_MULTILINE: (
            i32,
            i32
        );

        fn foo(&self) {
            todo!()
        }

        fn baz(&self) {
            todo!()
        }

    }
}"#,
        );
    }

    #[test]
    fn test_add_default_preserves_indentation() {
        check_assist(
            add_missing_default_members,
            r#"
mod m {
    pub trait Foo {
        type Output;

        const CONST: usize = 42;
        const CONST_2: i32;
        const CONST_MULTILINE: = (
            i32,
            i32,
        ) = (3, 14);

        fn valid(some: u32) -> bool { false }
        fn foo(some: u32) -> bool;
    }
}
struct S;
impl m::Foo for S { $0 }"#,
            r#"
mod m {
    pub trait Foo {
        type Output;

        const CONST: usize = 42;
        const CONST_2: i32;
        const CONST_MULTILINE: = (
            i32,
            i32,
        ) = (3, 14);

        fn valid(some: u32) -> bool { false }
        fn foo(some: u32) -> bool;
    }
}
struct S;
impl m::Foo for S {
    $0const CONST: usize = 42;

    const CONST_MULTILINE: = (
        i32,
        i32,
    ) = (3, 14);

    fn valid(some: u32) -> bool { false }
}"#,
        )
    }

    #[test]
    fn nested_macro_should_not_cause_crash() {
        check_assist(
            add_missing_impl_members,
            r#"
macro_rules! ty { () => { i32 } }
trait SomeTrait { type Output; }
impl SomeTrait for i32 { type Output = i64; }
macro_rules! define_method {
    () => {
        fn method(&mut self, params: <ty!() as SomeTrait>::Output);
    };
}
trait AnotherTrait { define_method!(); }
impl $0AnotherTrait for () {
}
"#,
            r#"
macro_rules! ty { () => { i32 } }
trait SomeTrait { type Output; }
impl SomeTrait for i32 { type Output = i64; }
macro_rules! define_method {
    () => {
        fn method(&mut self, params: <ty!() as SomeTrait>::Output);
    };
}
trait AnotherTrait { define_method!(); }
impl AnotherTrait for () {
    $0fn method(&mut self,params: <ty!()as SomeTrait>::Output) {
        todo!()
    }
}
"#,
        );
    }

    // FIXME: `T` in `ty!(T)` should be replaced by `PathTransform`.
    #[test]
    fn paths_in_nested_macro_should_get_transformed() {
        check_assist(
            add_missing_impl_members,
            r#"
macro_rules! ty { ($me:ty) => { $me } }
trait SomeTrait { type Output; }
impl SomeTrait for i32 { type Output = i64; }
macro_rules! define_method {
    ($t:ty) => {
        fn method(&mut self, params: <ty!($t) as SomeTrait>::Output);
    };
}
trait AnotherTrait<T: SomeTrait> { define_method!(T); }
impl $0AnotherTrait<i32> for () {
}
"#,
            r#"
macro_rules! ty { ($me:ty) => { $me } }
trait SomeTrait { type Output; }
impl SomeTrait for i32 { type Output = i64; }
macro_rules! define_method {
    ($t:ty) => {
        fn method(&mut self, params: <ty!($t) as SomeTrait>::Output);
    };
}
trait AnotherTrait<T: SomeTrait> { define_method!(T); }
impl AnotherTrait<i32> for () {
    $0fn method(&mut self,params: <ty!(T)as SomeTrait>::Output) {
        todo!()
    }
}
"#,
        );
    }
}
