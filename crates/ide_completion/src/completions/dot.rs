//! Completes references after dot (fields and method calls).

use hir::{HasVisibility, Type};
use rustc_hash::FxHashSet;

use crate::{context::CompletionContext, Completions};

/// Complete dot accesses, i.e. fields or methods.
pub(crate) fn complete_dot(acc: &mut Completions, ctx: &CompletionContext) {
    let dot_receiver = match &ctx.dot_receiver {
        Some(expr) => expr,
        _ => return,
    };

    let receiver_ty = match ctx.sema.type_of_expr(&dot_receiver) {
        Some(ty) => ty,
        _ => return,
    };

    if ctx.is_call {
        cov_mark::hit!(test_no_struct_field_completion_for_method_call);
    } else {
        complete_fields(acc, ctx, &receiver_ty);
    }
    complete_methods(acc, ctx, &receiver_ty);
}

fn complete_fields(acc: &mut Completions, ctx: &CompletionContext, receiver: &Type) {
    for receiver in receiver.autoderef(ctx.db) {
        for (field, ty) in receiver.fields(ctx.db) {
            if ctx.scope.module().map_or(false, |m| !field.is_visible_from(ctx.db, m)) {
                // Skip private field. FIXME: If the definition location of the
                // field is editable, we should show the completion
                continue;
            }
            acc.add_field(ctx, field, &ty);
        }
        for (i, ty) in receiver.tuple_fields(ctx.db).into_iter().enumerate() {
            // FIXME: Handle visibility
            acc.add_tuple_field(ctx, i, &ty);
        }
    }
}

fn complete_methods(acc: &mut Completions, ctx: &CompletionContext, receiver: &Type) {
    if let Some(krate) = ctx.krate {
        let mut seen_methods = FxHashSet::default();
        let traits_in_scope = ctx.scope.traits_in_scope();
        receiver.iterate_method_candidates(ctx.db, krate, &traits_in_scope, None, |_ty, func| {
            if func.self_param(ctx.db).is_some()
                && ctx.scope.module().map_or(true, |m| func.is_visible_from(ctx.db, m))
                && seen_methods.insert(func.name(ctx.db))
            {
                acc.add_function(ctx, func, None);
            }
            None::<()>
        });
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
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
            expect![[""]],
        );
    }

    #[test]
    fn test_visibility_filtering() {
        check(
            r#"
mod inner {
    pub struct A {
        private_field: u32,
        pub pub_field: u32,
        pub(crate) crate_field: u32,
        pub(crate) super_field: u32,
    }
}
fn foo(a: inner::A) { a.$0 }
"#,
            expect![[r#"
                fd pub_field   u32
                fd crate_field u32
                fd super_field u32
            "#]],
        );

        check(
            r#"
struct A {}
mod m {
    impl super::A {
        fn private_method(&self) {}
        pub(crate) fn the_method(&self) {}
    }
}
fn foo(a: A) { a.$0 }
"#,
            expect![[r#"
                me the_method() fn(&self)
            "#]],
        );
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
                me the_method() fn(&self)
            "#]],
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
                me the_method() fn(&self)
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
                me the_method() fn(&self)
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
            expect![[""]],
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
        )
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
}
