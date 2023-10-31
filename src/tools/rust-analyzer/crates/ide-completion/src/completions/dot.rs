//! Completes references after dot (fields and method calls).

use ide_db::FxHashSet;

use crate::{
    context::{CompletionContext, DotAccess, DotAccessKind, ExprCtx, PathCompletionCtx, Qualified},
    CompletionItem, CompletionItemKind, Completions,
};

/// Complete dot accesses, i.e. fields or methods.
pub(crate) fn complete_dot(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess,
) {
    let receiver_ty = match dot_access {
        DotAccess { receiver_ty: Some(receiver_ty), .. } => &receiver_ty.original,
        _ => return,
    };

    // Suggest .await syntax for types that implement Future trait
    if receiver_ty.impls_into_future(ctx.db) {
        let mut item =
            CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), "await");
        item.detail("expr.await");
        item.add_to(acc, ctx.db);
    }

    if let DotAccessKind::Method { .. } = dot_access.kind {
        cov_mark::hit!(test_no_struct_field_completion_for_method_call);
    } else {
        complete_fields(
            acc,
            ctx,
            receiver_ty,
            |acc, field, ty| acc.add_field(ctx, dot_access, None, field, &ty),
            |acc, field, ty| acc.add_tuple_field(ctx, None, field, &ty),
        );
    }
    complete_methods(ctx, receiver_ty, |func| acc.add_method(ctx, dot_access, func, None, None));
}

pub(crate) fn complete_undotted_self(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx,
    expr_ctx: &ExprCtx,
) {
    if !ctx.config.enable_self_on_the_fly {
        return;
    }
    if !path_ctx.is_trivial_path() {
        return;
    }
    if !ctx.qualifier_ctx.none() {
        return;
    }
    if !matches!(path_ctx.qualified, Qualified::No) {
        return;
    }
    let self_param = match expr_ctx {
        ExprCtx { self_param: Some(self_param), .. } => self_param,
        _ => return,
    };

    let ty = self_param.ty(ctx.db);
    complete_fields(
        acc,
        ctx,
        &ty,
        |acc, field, ty| {
            acc.add_field(
                ctx,
                &DotAccess {
                    receiver: None,
                    receiver_ty: None,
                    kind: DotAccessKind::Field { receiver_is_ambiguous_float_literal: false },
                },
                Some(hir::known::SELF_PARAM),
                field,
                &ty,
            )
        },
        |acc, field, ty| acc.add_tuple_field(ctx, Some(hir::known::SELF_PARAM), field, &ty),
    );
    complete_methods(ctx, &ty, |func| {
        acc.add_method(
            ctx,
            &DotAccess {
                receiver: None,
                receiver_ty: None,
                kind: DotAccessKind::Method { has_parens: false },
            },
            func,
            Some(hir::known::SELF_PARAM),
            None,
        )
    });
}

fn complete_fields(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    receiver: &hir::Type,
    mut named_field: impl FnMut(&mut Completions, hir::Field, hir::Type),
    mut tuple_index: impl FnMut(&mut Completions, usize, hir::Type),
) {
    let mut seen_names = FxHashSet::default();
    for receiver in receiver.autoderef(ctx.db) {
        for (field, ty) in receiver.fields(ctx.db) {
            if seen_names.insert(field.name(ctx.db)) {
                named_field(acc, field, ty);
            }
        }
        for (i, ty) in receiver.tuple_fields(ctx.db).into_iter().enumerate() {
            // Tuples are always the last type in a deref chain, so just check if the name is
            // already seen without inserting into the hashset.
            if !seen_names.contains(&hir::Name::new_tuple_field(i)) {
                // Tuple fields are always public (tuple struct fields are handled above).
                tuple_index(acc, i, ty);
            }
        }
    }
}

fn complete_methods(
    ctx: &CompletionContext<'_>,
    receiver: &hir::Type,
    mut f: impl FnMut(hir::Function),
) {
    let mut seen_methods = FxHashSet::default();
    receiver.iterate_method_candidates_with_traits(
        ctx.db,
        &ctx.scope,
        &ctx.traits_in_scope(),
        Some(ctx.module),
        None,
        |func| {
            if func.self_param(ctx.db).is_some() && seen_methods.insert(func.name(ctx.db)) {
                f(func);
            }
            None::<()>
        },
    );
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::{
        check_edit, completion_list_no_kw, completion_list_no_kw_with_private_editable,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list_no_kw(ra_fixture);
        expect.assert_eq(&actual);
    }

    fn check_with_private_editable(ra_fixture: &str, expect: Expect) {
        let actual = completion_list_no_kw_with_private_editable(ra_fixture);
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_struct_field_and_method_completion() {
        check(
            r#"
struct S { foo: u32 }
impl S {
    fn bar(&self) {}
}
fn foo(s: S) { s.$0 }
"#,
            expect![[r#"
                fd foo   u32
                me bar() fn(&self)
            "#]],
        );
    }

    #[test]
    fn no_unstable_method_on_stable() {
        check(
            r#"
//- /main.rs crate:main deps:std
fn foo(s: std::S) { s.$0 }
//- /std.rs crate:std
pub struct S;
impl S {
    #[unstable]
    pub fn bar(&self) {}
}
"#,
            expect![""],
        );
    }

    #[test]
    fn unstable_method_on_nightly() {
        check(
            r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
fn foo(s: std::S) { s.$0 }
//- /std.rs crate:std
pub struct S;
impl S {
    #[unstable]
    pub fn bar(&self) {}
}
"#,
            expect![[r#"
                me bar() fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_struct_field_completion_self() {
        check(
            r#"
struct S { the_field: (u32,) }
impl S {
    fn foo(self) { self.$0 }
}
"#,
            expect![[r#"
                fd the_field (u32,)
                me foo()     fn(self)
            "#]],
        )
    }

    #[test]
    fn test_struct_field_completion_autoderef() {
        check(
            r#"
struct A { the_field: (u32, i32) }
impl A {
    fn foo(&self) { self.$0 }
}
"#,
            expect![[r#"
                fd the_field (u32, i32)
                me foo()     fn(&self)
            "#]],
        )
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        cov_mark::check!(test_no_struct_field_completion_for_method_call);
        check(
            r#"
struct A { the_field: u32 }
fn foo(a: A) { a.$0() }
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn test_visibility_filtering() {
        check(
            r#"
//- /lib.rs crate:lib new_source_root:local
pub mod m {
    pub struct A {
        private_field: u32,
        pub pub_field: u32,
        pub(crate) crate_field: u32,
        pub(super) super_field: u32,
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd pub_field u32
            "#]],
        );

        check(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub mod m {
    pub struct A {
        private_field: u32,
        pub pub_field: u32,
        pub(crate) crate_field: u32,
        pub(super) super_field: u32,
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd pub_field u32
            "#]],
        );

        check(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub mod m {
    pub struct A(
        i32,
        pub f64,
    );
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd 1 f64
            "#]],
        );

        check(
            r#"
//- /lib.rs crate:lib new_source_root:local
pub struct A {}
mod m {
    impl super::A {
        fn private_method(&self) {}
        pub(crate) fn crate_method(&self) {}
        pub fn pub_method(&self) {}
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::A) { a.$0 }
"#,
            expect![[r#"
                me pub_method() fn(&self)
            "#]],
        );
        check(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub struct A {}
mod m {
    impl super::A {
        fn private_method(&self) {}
        pub(crate) fn crate_method(&self) {}
        pub fn pub_method(&self) {}
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::A) { a.$0 }
"#,
            expect![[r#"
                me pub_method() fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_visibility_filtering_with_private_editable_enabled() {
        check_with_private_editable(
            r#"
//- /lib.rs crate:lib new_source_root:local
pub mod m {
    pub struct A {
        private_field: u32,
        pub pub_field: u32,
        pub(crate) crate_field: u32,
        pub(super) super_field: u32,
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd crate_field   u32
                fd private_field u32
                fd pub_field     u32
                fd super_field   u32
            "#]],
        );

        check_with_private_editable(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub mod m {
    pub struct A {
        private_field: u32,
        pub pub_field: u32,
        pub(crate) crate_field: u32,
        pub(super) super_field: u32,
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd pub_field u32
            "#]],
        );

        check_with_private_editable(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub mod m {
    pub struct A(
        i32,
        pub f64,
    );
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::m::A) { a.$0 }
"#,
            expect![[r#"
                fd 1 f64
            "#]],
        );

        check_with_private_editable(
            r#"
//- /lib.rs crate:lib new_source_root:local
pub struct A {}
mod m {
    impl super::A {
        fn private_method(&self) {}
        pub(crate) fn crate_method(&self) {}
        pub fn pub_method(&self) {}
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::A) { a.$0 }
"#,
            expect![[r#"
                me crate_method()   fn(&self)
                me private_method() fn(&self)
                me pub_method()     fn(&self)
            "#]],
        );
        check_with_private_editable(
            r#"
//- /lib.rs crate:lib new_source_root:library
pub struct A {}
mod m {
    impl super::A {
        fn private_method(&self) {}
        pub(crate) fn crate_method(&self) {}
        pub fn pub_method(&self) {}
    }
}
//- /main.rs crate:main deps:lib new_source_root:local
fn foo(a: lib::A) { a.$0 }
"#,
            expect![[r#"
                me pub_method() fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_local_impls() {
        check(
            r#"
pub struct A {}
mod m {
    impl super::A {
        pub fn pub_module_method(&self) {}
    }
    fn f() {
        impl super::A {
            pub fn pub_foreign_local_method(&self) {}
        }
    }
}
fn foo(a: A) {
    impl A {
        fn local_method(&self) {}
    }
    a.$0
}
"#,
            expect![[r#"
                me local_method()      fn(&self)
                me pub_module_method() fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_doc_hidden_filtering() {
        check(
            r#"
//- /lib.rs crate:lib deps:dep
fn foo(a: dep::A) { a.$0 }
//- /dep.rs crate:dep
pub struct A {
    #[doc(hidden)]
    pub hidden_field: u32,
    pub pub_field: u32,
}

impl A {
    pub fn pub_method(&self) {}

    #[doc(hidden)]
    pub fn hidden_method(&self) {}
}
            "#,
            expect![[r#"
                fd pub_field    u32
                me pub_method() fn(&self)
            "#]],
        )
    }

    #[test]
    fn test_union_field_completion() {
        check(
            r#"
union U { field: u8, other: u16 }
fn foo(u: U) { u.$0 }
"#,
            expect![[r#"
                fd field u8
                fd other u16
            "#]],
        );
    }

    #[test]
    fn test_method_completion_only_fitting_impls() {
        check(
            r#"
struct A<T> {}
impl A<u32> {
    fn the_method(&self) {}
}
impl A<i32> {
    fn the_other_method(&self) {}
}
fn foo(a: A<u32>) { a.$0 }
"#,
            expect![[r#"
                me the_method() fn(&self)
            "#]],
        )
    }

    #[test]
    fn test_trait_method_completion() {
        check(
            r#"
struct A {}
trait Trait { fn the_method(&self); }
impl Trait for A {}
fn foo(a: A) { a.$0 }
"#,
            expect![[r#"
                me the_method() (as Trait) fn(&self)
            "#]],
        );
        check_edit(
            "the_method",
            r#"
struct A {}
trait Trait { fn the_method(&self); }
impl Trait for A {}
fn foo(a: A) { a.$0 }
"#,
            r#"
struct A {}
trait Trait { fn the_method(&self); }
impl Trait for A {}
fn foo(a: A) { a.the_method()$0 }
"#,
        );
    }

    #[test]
    fn test_trait_method_completion_deduplicated() {
        check(
            r"
struct A {}
trait Trait { fn the_method(&self); }
impl<T> Trait for T {}
fn foo(a: &A) { a.$0 }
",
            expect![[r#"
                me the_method() (as Trait) fn(&self)
            "#]],
        );
    }

    #[test]
    fn completes_trait_method_from_other_module() {
        check(
            r"
struct A {}
mod m {
    pub trait Trait { fn the_method(&self); }
}
use m::Trait;
impl Trait for A {}
fn foo(a: A) { a.$0 }
",
            expect![[r#"
                me the_method() (as Trait) fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_no_non_self_method() {
        check(
            r#"
struct A {}
impl A {
    fn the_method() {}
}
fn foo(a: A) {
   a.$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn test_tuple_field_completion() {
        check(
            r#"
fn foo() {
   let b = (0, 3.14);
   b.$0
}
"#,
            expect![[r#"
                fd 0 i32
                fd 1 f64
            "#]],
        );
    }

    #[test]
    fn test_tuple_struct_field_completion() {
        check(
            r#"
struct S(i32, f64);
fn foo() {
   let b = S(0, 3.14);
   b.$0
}
"#,
            expect![[r#"
                fd 0 i32
                fd 1 f64
            "#]],
        );
    }

    #[test]
    fn test_tuple_field_inference() {
        check(
            r#"
pub struct S;
impl S { pub fn blah(&self) {} }

struct T(S);

impl T {
    fn foo(&self) {
        // FIXME: This doesn't work without the trailing `a` as `0.` is a float
        self.0.a$0
    }
}
"#,
            expect![[r#"
                me blah() fn(&self)
            "#]],
        );
    }

    #[test]
    fn test_field_no_same_name() {
        check(
            r#"
//- minicore: deref
struct A { field: u8 }
struct B { field: u16, another: u32 }
impl core::ops::Deref for A {
    type Target = B;
    fn deref(&self) -> &Self::Target { loop {} }
}
fn test(a: A) {
    a.$0
}
"#,
            expect![[r#"
                fd another                u32
                fd field                  u8
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_tuple_field_no_same_index() {
        check(
            r#"
//- minicore: deref
struct A(u8);
struct B(u16, u32);
impl core::ops::Deref for A {
    type Target = B;
    fn deref(&self) -> &Self::Target { loop {} }
}
fn test(a: A) {
    a.$0
}
"#,
            expect![[r#"
                fd 0                      u8
                fd 1                      u32
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_tuple_struct_deref_to_tuple_no_same_index() {
        check(
            r#"
//- minicore: deref
struct A(u8);
impl core::ops::Deref for A {
    type Target = (u16, u32);
    fn deref(&self) -> &Self::Target { loop {} }
}
fn test(a: A) {
    a.$0
}
"#,
            expect![[r#"
                fd 0                      u8
                fd 1                      u32
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_completion_works_in_consts() {
        check(
            r#"
struct A { the_field: u32 }
const X: u32 = {
    A { the_field: 92 }.$0
};
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn works_in_simple_macro_1() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
struct A { the_field: u32 }
fn foo(a: A) {
    m!(a.x$0)
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn works_in_simple_macro_2() {
        // this doesn't work yet because the macro doesn't expand without the token -- maybe it can be fixed with better recovery
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
struct A { the_field: u32 }
fn foo(a: A) {
    m!(a.$0)
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn works_in_simple_macro_recursive_1() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
struct A { the_field: u32 }
fn foo(a: A) {
    m!(m!(m!(a.x$0)))
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn macro_expansion_resilient() {
        check(
            r#"
macro_rules! d {
    () => {};
    ($val:expr) => {
        match $val { tmp => { tmp } }
    };
    // Trailing comma with single argument is ignored
    ($val:expr,) => { $crate::d!($val) };
    ($($val:expr),+ $(,)?) => {
        ($($crate::d!($val)),+,)
    };
}
struct A { the_field: u32 }
fn foo(a: A) {
    d!(a.$0)
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn test_method_completion_issue_3547() {
        check(
            r#"
struct HashSet<T> {}
impl<T> HashSet<T> {
    pub fn the_method(&self) {}
}
fn foo() {
    let s: HashSet<_>;
    s.$0
}
"#,
            expect![[r#"
                me the_method() fn(&self)
            "#]],
        );
    }

    #[test]
    fn completes_method_call_when_receiver_is_a_macro_call() {
        check(
            r#"
struct S;
impl S { fn foo(&self) {} }
macro_rules! make_s { () => { S }; }
fn main() { make_s!().f$0; }
"#,
            expect![[r#"
                me foo() fn(&self)
            "#]],
        )
    }

    #[test]
    fn completes_after_macro_call_in_submodule() {
        check(
            r#"
macro_rules! empty {
    () => {};
}

mod foo {
    #[derive(Debug, Default)]
    struct Template2 {}

    impl Template2 {
        fn private(&self) {}
    }
    fn baz() {
        let goo: Template2 = Template2 {};
        empty!();
        goo.$0
    }
}
        "#,
            expect![[r#"
                me private() fn(&self)
            "#]],
        );
    }

    #[test]
    fn issue_8931() {
        check(
            r#"
//- minicore: fn
struct S;

struct Foo;
impl Foo {
    fn foo(&self) -> &[u8] { loop {} }
}

impl S {
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
    }

    fn f(&mut self, v: Foo) {
        self.indented(|this| v.$0)
    }
}
        "#,
            expect![[r#"
                me foo() fn(&self) -> &[u8]
            "#]],
        );
    }

    #[test]
    fn completes_bare_fields_and_methods_in_methods() {
        check(
            r#"
struct Foo { field: i32 }

impl Foo { fn foo(&self) { $0 } }"#,
            expect![[r#"
                fd self.field i32
                lc self       &Foo
                sp Self
                st Foo
                bt u32
                me self.foo() fn(&self)
            "#]],
        );
        check(
            r#"
struct Foo(i32);

impl Foo { fn foo(&mut self) { $0 } }"#,
            expect![[r#"
                fd self.0     i32
                lc self       &mut Foo
                sp Self
                st Foo
                bt u32
                me self.foo() fn(&mut self)
            "#]],
        );
    }

    #[test]
    fn macro_completion_after_dot() {
        check(
            r#"
macro_rules! m {
    ($e:expr) => { $e };
}

struct Completable;

impl Completable {
    fn method(&self) {}
}

fn f() {
    let c = Completable;
    m!(c.$0);
}
    "#,
            expect![[r#"
                me method() fn(&self)
            "#]],
        );
    }

    #[test]
    fn completes_method_call_when_receiver_type_has_errors_issue_10297() {
        check(
            r#"
//- minicore: iterator, sized
struct Vec<T>;
impl<T> IntoIterator for Vec<T> {
    type Item = ();
    type IntoIter = ();
    fn into_iter(self);
}
fn main() {
    let x: Vec<_>;
    x.$0;
}
"#,
            expect![[r#"
                me into_iter() (as IntoIterator) fn(self) -> <Self as IntoIterator>::IntoIter
            "#]],
        )
    }

    #[test]
    fn postfix_drop_completion() {
        cov_mark::check!(postfix_drop_completion);
        check_edit(
            "drop",
            r#"
//- minicore: drop
struct Vec<T>(T);
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {}
}
fn main() {
    let x = Vec(0u32)
    x.$0;
}
"#,
            r"
struct Vec<T>(T);
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {}
}
fn main() {
    let x = Vec(0u32)
    drop($0x);
}
",
        )
    }

    #[test]
    fn issue_12484() {
        check(
            r#"
//- minicore: sized
trait SizeUser {
    type Size;
}
trait Closure: SizeUser {}
trait Encrypt: SizeUser {
    fn encrypt(self, _: impl Closure<Size = Self::Size>);
}
fn test(thing: impl Encrypt) {
    thing.$0;
}
        "#,
            expect![[r#"
                me encrypt(…) (as Encrypt) fn(self, impl Closure<Size = <Self as SizeUser>::Size>)
            "#]],
        )
    }

    #[test]
    fn only_consider_same_type_once() {
        check(
            r#"
//- minicore: deref
struct A(u8);
struct B(u16);
impl core::ops::Deref for A {
    type Target = B;
    fn deref(&self) -> &Self::Target { loop {} }
}
impl core::ops::Deref for B {
    type Target = A;
    fn deref(&self) -> &Self::Target { loop {} }
}
fn test(a: A) {
    a.$0
}
"#,
            expect![[r#"
                fd 0                      u8
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn no_inference_var_in_completion() {
        check(
            r#"
struct S<T>(T);
fn test(s: S<Unknown>) {
    s.$0
}
"#,
            expect![[r#"
                fd 0 {unknown}
            "#]],
        );
    }
}
