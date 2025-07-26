//! Completes references after dot (fields and method calls).

use std::ops::ControlFlow;

use hir::{Complete, HasContainer, ItemContainer, MethodCandidateCallback, Name};
use ide_db::FxHashSet;
use syntax::SmolStr;

use crate::{
    CompletionItem, CompletionItemKind, Completions,
    context::{
        CompletionContext, DotAccess, DotAccessExprCtx, DotAccessKind, PathCompletionCtx,
        PathExprCtx, Qualified,
    },
};

/// Complete dot accesses, i.e. fields or methods.
pub(crate) fn complete_dot(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess<'_>,
) {
    let receiver_ty = match dot_access {
        DotAccess { receiver_ty: Some(receiver_ty), .. } => &receiver_ty.original,
        _ => return,
    };

    let is_field_access = matches!(dot_access.kind, DotAccessKind::Field { .. });
    let is_method_access_with_parens =
        matches!(dot_access.kind, DotAccessKind::Method { has_parens: true });
    let traits_in_scope = ctx.traits_in_scope();

    // Suggest .await syntax for types that implement Future trait
    if let Some(future_output) = receiver_ty.into_future_output(ctx.db) {
        let await_str = SmolStr::new_static("await");
        let mut item = CompletionItem::new(
            CompletionItemKind::Keyword,
            ctx.source_range(),
            await_str.clone(),
            ctx.edition,
        );
        item.detail("expr.await");
        item.add_to(acc, ctx.db);

        if ctx.config.enable_auto_await {
            // Completions that skip `.await`, e.g. `.await.foo()`.
            let dot_access_kind = match &dot_access.kind {
                DotAccessKind::Field { receiver_is_ambiguous_float_literal: _ } => {
                    DotAccessKind::Field { receiver_is_ambiguous_float_literal: false }
                }
                it @ DotAccessKind::Method { .. } => *it,
            };
            let dot_access = DotAccess {
                receiver: dot_access.receiver.clone(),
                receiver_ty: Some(hir::TypeInfo {
                    original: future_output.clone(),
                    adjusted: None,
                }),
                kind: dot_access_kind,
                ctx: dot_access.ctx,
            };
            complete_fields(
                acc,
                ctx,
                &future_output,
                |acc, field, ty| {
                    acc.add_field(ctx, &dot_access, Some(await_str.clone()), field, &ty)
                },
                |acc, field, ty| acc.add_tuple_field(ctx, Some(await_str.clone()), field, &ty),
                is_field_access,
                is_method_access_with_parens,
            );
            complete_methods(ctx, &future_output, &traits_in_scope, |func| {
                acc.add_method(ctx, &dot_access, func, Some(await_str.clone()), None)
            });
        }
    }

    complete_fields(
        acc,
        ctx,
        receiver_ty,
        |acc, field, ty| acc.add_field(ctx, dot_access, None, field, &ty),
        |acc, field, ty| acc.add_tuple_field(ctx, None, field, &ty),
        is_field_access,
        is_method_access_with_parens,
    );
    complete_methods(ctx, receiver_ty, &traits_in_scope, |func| {
        acc.add_method(ctx, dot_access, func, None, None)
    });

    if ctx.config.enable_auto_iter && !receiver_ty.strip_references().impls_iterator(ctx.db) {
        // FIXME:
        // Checking for the existence of `iter()` is complicated in our setup, because we need to substitute
        // its return type, so we instead check for `<&Self as IntoIterator>::IntoIter`.
        // Does <&receiver_ty as IntoIterator>::IntoIter` exist? Assume `iter` is valid
        let iter = receiver_ty
            .strip_references()
            .add_reference(hir::Mutability::Shared)
            .into_iterator_iter(ctx.db)
            .map(|ty| (ty, SmolStr::new_static("iter()")));
        // Does <receiver_ty as IntoIterator>::IntoIter` exist?
        let into_iter = || {
            receiver_ty
                .clone()
                .into_iterator_iter(ctx.db)
                .map(|ty| (ty, SmolStr::new_static("into_iter()")))
        };
        if let Some((iter, iter_sym)) = iter.or_else(into_iter) {
            // Skip iterators, e.g. complete `.iter().filter_map()`.
            let dot_access_kind = match &dot_access.kind {
                DotAccessKind::Field { receiver_is_ambiguous_float_literal: _ } => {
                    DotAccessKind::Field { receiver_is_ambiguous_float_literal: false }
                }
                it @ DotAccessKind::Method { .. } => *it,
            };
            let dot_access = DotAccess {
                receiver: dot_access.receiver.clone(),
                receiver_ty: Some(hir::TypeInfo { original: iter.clone(), adjusted: None }),
                kind: dot_access_kind,
                ctx: dot_access.ctx,
            };
            complete_methods(ctx, &iter, &traits_in_scope, |func| {
                acc.add_method(ctx, &dot_access, func, Some(iter_sym.clone()), None)
            });
        }
    }
}

pub(crate) fn complete_undotted_self(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    expr_ctx: &PathExprCtx<'_>,
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
        PathExprCtx { self_param: Some(self_param), .. } => self_param,
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
                    ctx: DotAccessExprCtx {
                        in_block_expr: expr_ctx.in_block_expr,
                        in_breakable: expr_ctx.in_breakable,
                    },
                },
                Some(SmolStr::new_static("self")),
                field,
                &ty,
            )
        },
        |acc, field, ty| acc.add_tuple_field(ctx, Some(SmolStr::new_static("self")), field, &ty),
        true,
        false,
    );
    complete_methods(ctx, &ty, &ctx.traits_in_scope(), |func| {
        acc.add_method(
            ctx,
            &DotAccess {
                receiver: None,
                receiver_ty: None,
                kind: DotAccessKind::Method { has_parens: false },
                ctx: DotAccessExprCtx {
                    in_block_expr: expr_ctx.in_block_expr,
                    in_breakable: expr_ctx.in_breakable,
                },
            },
            func,
            Some(SmolStr::new_static("self")),
            None,
        )
    });
}

fn complete_fields(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    receiver: &hir::Type<'_>,
    mut named_field: impl FnMut(&mut Completions, hir::Field, hir::Type<'_>),
    mut tuple_index: impl FnMut(&mut Completions, usize, hir::Type<'_>),
    is_field_access: bool,
    is_method_access_with_parens: bool,
) {
    let mut seen_names = FxHashSet::default();
    for receiver in receiver.autoderef(ctx.db) {
        for (field, ty) in receiver.fields(ctx.db) {
            if seen_names.insert(field.name(ctx.db))
                && (is_field_access
                    || (is_method_access_with_parens && (ty.is_fn() || ty.is_closure())))
            {
                named_field(acc, field, ty);
            }
        }
        for (i, ty) in receiver.tuple_fields(ctx.db).into_iter().enumerate() {
            // Tuples are always the last type in a deref chain, so just check if the name is
            // already seen without inserting into the hashset.
            if !seen_names.contains(&hir::Name::new_tuple_field(i))
                && (is_field_access
                    || (is_method_access_with_parens && (ty.is_fn() || ty.is_closure())))
            {
                // Tuple fields are always public (tuple struct fields are handled above).
                tuple_index(acc, i, ty);
            }
        }
    }
}

fn complete_methods(
    ctx: &CompletionContext<'_>,
    receiver: &hir::Type<'_>,
    traits_in_scope: &FxHashSet<hir::TraitId>,
    f: impl FnMut(hir::Function),
) {
    struct Callback<'a, F> {
        ctx: &'a CompletionContext<'a>,
        f: F,
        seen_methods: FxHashSet<Name>,
    }

    impl<F> MethodCandidateCallback for Callback<'_, F>
    where
        F: FnMut(hir::Function),
    {
        // We don't want to exclude inherent trait methods - that is, methods of traits available from
        // `where` clauses or `dyn Trait`.
        fn on_inherent_method(&mut self, func: hir::Function) -> ControlFlow<()> {
            if func.self_param(self.ctx.db).is_some()
                && self.seen_methods.insert(func.name(self.ctx.db))
            {
                (self.f)(func);
            }
            ControlFlow::Continue(())
        }

        fn on_trait_method(&mut self, func: hir::Function) -> ControlFlow<()> {
            // This needs to come before the `seen_methods` test, so that if we see the same method twice,
            // once as inherent and once not, we will include it.
            if let ItemContainer::Trait(trait_) = func.container(self.ctx.db) {
                if self.ctx.exclude_traits.contains(&trait_)
                    || trait_.complete(self.ctx.db) == Complete::IgnoreMethods
                {
                    return ControlFlow::Continue(());
                }
            }

            if func.self_param(self.ctx.db).is_some()
                && self.seen_methods.insert(func.name(self.ctx.db))
            {
                (self.f)(func);
            }

            ControlFlow::Continue(())
        }
    }

    receiver.iterate_method_candidates_split_inherent(
        ctx.db,
        &ctx.scope,
        traits_in_scope,
        Some(ctx.module),
        None,
        Callback { ctx, f, seen_methods: FxHashSet::default() },
    );
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::{check_edit, check_no_kw, check_with_private_editable};

    #[test]
    fn test_struct_field_and_method_completion() {
        check_no_kw(
            r#"
struct S { foo: u32 }
impl S {
    fn bar(&self) {}
}
fn foo(s: S) { s.$0 }
"#,
            expect![[r#"
                fd foo         u32
                me bar() fn(&self)
            "#]],
        );
    }

    #[test]
    fn no_unstable_method_on_stable() {
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
            r#"
struct S { the_field: (u32,) }
impl S {
    fn foo(self) { self.$0 }
}
"#,
            expect![[r#"
                fd the_field (u32,)
                me foo()   fn(self)
            "#]],
        )
    }

    #[test]
    fn test_struct_field_completion_autoderef() {
        check_no_kw(
            r#"
struct A { the_field: (u32, i32) }
impl A {
    fn foo(&self) { self.$0 }
}
"#,
            expect![[r#"
                fd the_field (u32, i32)
                me foo()      fn(&self)
            "#]],
        )
    }

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        check_no_kw(
            r#"
struct A { the_field: u32 }
fn foo(a: A) { a.$0() }
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn test_visibility_filtering() {
        check_no_kw(
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

        check_no_kw(
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

        check_no_kw(
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

        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
                fd pub_field          u32
                me pub_method() fn(&self)
            "#]],
        )
    }

    #[test]
    fn test_union_field_completion() {
        check_no_kw(
            r#"
union U { field: u8, other: u16 }
fn foo(u: U) { u.$0 }
"#,
            expect![[r#"
                fd field  u8
                fd other u16
            "#]],
        );
    }

    #[test]
    fn test_method_completion_only_fitting_impls() {
        check_no_kw(
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
        check_no_kw(
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
fn foo(a: A) { a.the_method();$0 }
"#,
        );
    }

    #[test]
    fn test_trait_method_completion_deduplicated() {
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
                fd another                                                          u32
                fd field                                                             u8
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_tuple_field_no_same_index() {
        check_no_kw(
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
                fd 0                                                                 u8
                fd 1                                                                u32
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_tuple_struct_deref_to_tuple_no_same_index() {
        check_no_kw(
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
                fd 0                                                                 u8
                fd 1                                                                u32
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn test_completion_works_in_consts() {
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
            r#"
struct Foo { field: i32 }

impl Foo { fn foo(&self) { $0 } }"#,
            expect![[r#"
                fd self.field       i32
                me self.foo() fn(&self)
                lc self            &Foo
                sp Self             Foo
                st Foo              Foo
                bt u32              u32
            "#]],
        );
        check_no_kw(
            r#"
struct Foo(i32);

impl Foo { fn foo(&mut self) { $0 } }"#,
            expect![[r#"
                fd self.0               i32
                me self.foo() fn(&mut self)
                lc self            &mut Foo
                sp Self                 Foo
                st Foo                  Foo
                bt u32                  u32
            "#]],
        );
    }

    #[test]
    fn macro_completion_after_dot() {
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
        check_no_kw(
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
                fd 0                                                                 u8
                me deref() (use core::ops::Deref) fn(&self) -> &<Self as Deref>::Target
            "#]],
        );
    }

    #[test]
    fn no_inference_var_in_completion() {
        check_no_kw(
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

    #[test]
    fn assoc_impl_1() {
        check_no_kw(
            r#"
//- minicore: deref
fn main() {
    let foo: Foo<&u8> = Foo::new(&42_u8);
    foo.$0
}

trait Bar {
    fn bar(&self);
}

impl Bar for u8 {
    fn bar(&self) {}
}

struct Foo<F> {
    foo: F,
}

impl<F> Foo<F> {
    fn new(foo: F) -> Foo<F> {
        Foo { foo }
    }
}

impl<F: core::ops::Deref<Target = impl Bar>> Foo<F> {
    fn foobar(&self) {
        self.foo.deref().bar()
    }
}
"#,
            expect![[r#"
                fd foo            &u8
                me foobar() fn(&self)
            "#]],
        );
    }

    #[test]
    fn assoc_impl_2() {
        check_no_kw(
            r#"
//- minicore: deref
fn main() {
    let foo: Foo<&u8> = Foo::new(&42_u8);
    foo.$0
}

trait Bar {
    fn bar(&self);
}

struct Foo<F> {
    foo: F,
}

impl<F> Foo<F> {
    fn new(foo: F) -> Foo<F> {
        Foo { foo }
    }
}

impl<B: Bar, F: core::ops::Deref<Target = B>> Foo<F> {
    fn foobar(&self) {
        self.foo.deref().bar()
    }
}
"#,
            expect![[r#"
                fd foo &u8
            "#]],
        );
    }

    #[test]
    fn test_struct_function_field_completion() {
        check_no_kw(
            r#"
struct S { va_field: u32, fn_field: fn() }
fn foo() { S { va_field: 0, fn_field: || {} }.fi$0() }
"#,
            expect![[r#"
                fd fn_field fn()
            "#]],
        );

        check_edit(
            "fn_field",
            r#"
struct S { va_field: u32, fn_field: fn() }
fn foo() { S { va_field: 0, fn_field: || {} }.fi$0() }
"#,
            r#"
struct S { va_field: u32, fn_field: fn() }
fn foo() { (S { va_field: 0, fn_field: || {} }.fn_field)() }
"#,
        );
    }

    #[test]
    fn test_tuple_function_field_completion() {
        check_no_kw(
            r#"
struct B(u32, fn())
fn foo() {
   let b = B(0, || {});
   b.$0()
}
"#,
            expect![[r#"
                fd 1 fn()
            "#]],
        );

        check_edit(
            "1",
            r#"
struct B(u32, fn())
fn foo() {
   let b = B(0, || {});
   b.$0()
}
"#,
            r#"
struct B(u32, fn())
fn foo() {
   let b = B(0, || {});
   (b.1)()
}
"#,
        )
    }

    #[test]
    fn test_fn_field_dot_access_method_has_parens_false() {
        check_no_kw(
            r#"
struct Foo { baz: fn() }
impl Foo {
    fn bar<T>(self, t: T): T { t }
}

fn baz() {
    let foo = Foo{ baz: || {} };
    foo.ba$0::<>;
}
"#,
            expect![[r#"
                me bar(…) fn(self, T)
            "#]],
        );
    }

    #[test]
    fn skip_iter() {
        check_no_kw(
            r#"
        //- minicore: iterator
        fn foo() {
            [].$0
        }
        "#,
            expect![[r#"
                me clone() (as Clone)                                       fn(&self) -> Self
                me into_iter() (as IntoIterator) fn(self) -> <Self as IntoIterator>::IntoIter
            "#]],
        );
        check_no_kw(
            r#"
//- minicore: iterator
struct MyIntoIter;
impl IntoIterator for MyIntoIter {
    type Item = ();
    type IntoIter = MyIterator;
    fn into_iter(self) -> Self::IntoIter {
        MyIterator
    }
}

struct MyIterator;
impl Iterator for MyIterator {
    type Item = ();
    fn next(&mut self) -> Self::Item {}
}

fn foo() {
    MyIntoIter.$0
}
"#,
            expect![[r#"
                me into_iter() (as IntoIterator)                fn(self) -> <Self as IntoIterator>::IntoIter
                me into_iter().by_ref() (as Iterator)                             fn(&mut self) -> &mut Self
                me into_iter().into_iter() (as IntoIterator)    fn(self) -> <Self as IntoIterator>::IntoIter
                me into_iter().next() (as Iterator)        fn(&mut self) -> Option<<Self as Iterator>::Item>
                me into_iter().nth(…) (as Iterator) fn(&mut self, usize) -> Option<<Self as Iterator>::Item>
            "#]],
        );
    }

    #[test]
    fn skip_await() {
        check_no_kw(
            r#"
//- minicore: future
struct Foo;
impl Foo {
    fn foo(self) {}
}

async fn foo() -> Foo { Foo }

async fn bar() {
    foo().$0
}
"#,
            expect![[r#"
    me await.foo()                                                                      fn(self)
    me into_future() (use core::future::IntoFuture) fn(self) -> <Self as IntoFuture>::IntoFuture
"#]],
        );
        check_edit(
            "foo",
            r#"
//- minicore: future
struct Foo;
impl Foo {
    fn foo(self) {}
}

async fn foo() -> Foo { Foo }

async fn bar() {
    foo().$0
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(self) {}
}

async fn foo() -> Foo { Foo }

async fn bar() {
    foo().await.foo();$0
}
"#,
        );
    }

    #[test]
    fn receiver_without_deref_impl_completion() {
        check_no_kw(
            r#"
//- minicore: receiver
use core::ops::Receiver;

struct Foo;

impl Foo {
    fn foo(self: Bar) {}
}

struct Bar;

impl Receiver for Bar {
    type Target = Foo;
}

fn main() {
    let bar = Bar;
    bar.$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn no_iter_suggestion_on_iterator() {
        check_no_kw(
            r#"
//- minicore: iterator
struct MyIter;
impl Iterator for MyIter {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> { None }
}

fn main() {
    MyIter.$0
}
"#,
            expect![[r#"
                me by_ref() (as Iterator)                             fn(&mut self) -> &mut Self
                me into_iter() (as IntoIterator)    fn(self) -> <Self as IntoIterator>::IntoIter
                me next() (as Iterator)        fn(&mut self) -> Option<<Self as Iterator>::Item>
                me nth(…) (as Iterator) fn(&mut self, usize) -> Option<<Self as Iterator>::Item>
            "#]],
        );
    }
}
