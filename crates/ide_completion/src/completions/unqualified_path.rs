//! Completion of names from the current scope, e.g. locals and imported items.

use hir::ScopeDef;
use syntax::AstNode;

use crate::{CompletionContext, Completions};

pub(crate) fn complete_unqualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path {
        return;
    }
    if ctx.record_lit_syntax.is_some()
        || ctx.record_pat_syntax.is_some()
        || ctx.attribute_under_caret.is_some()
        || ctx.mod_declaration_under_caret.is_some()
    {
        return;
    }

    if let Some(ty) = &ctx.expected_type {
        super::complete_enum_variants(acc, ctx, ty, |acc, ctx, variant, path| {
            acc.add_qualified_enum_variant(ctx, variant, path)
        });
    }

    ctx.scope.process_all_names(&mut |name, res| {
        if let ScopeDef::GenericParam(hir::GenericParam::LifetimeParam(_)) = res {
            cov_mark::hit!(skip_lifetime_completion);
            return;
        }
        if ctx.use_item_syntax.is_some() {
            if let (ScopeDef::Unknown, Some(name_ref)) = (&res, &ctx.name_ref_syntax) {
                if name_ref.syntax().text() == name.to_string().as_str() {
                    cov_mark::hit!(self_fulfilling_completion);
                    return;
                }
            }
        }
        acc.add_resolution(ctx, name.to_string(), &res);
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        test_utils::{check_edit, completion_list_with_config, TEST_CONFIG},
        CompletionConfig, CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        check_with_config(TEST_CONFIG, ra_fixture, expect);
    }

    fn check_with_config(config: CompletionConfig, ra_fixture: &str, expect: Expect) {
        let actual = completion_list_with_config(config, ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
    }

    #[test]
    fn self_fulfilling_completion() {
        cov_mark::check!(self_fulfilling_completion);
        check(
            r#"
use foo$0
use std::collections;
"#,
            expect![[r#"
                ?? collections
            "#]],
        );
    }

    #[test]
    fn bind_pat_and_path_ignore_at() {
        check(
            r#"
enum Enum { A, B }
fn quux(x: Option<Enum>) {
    match x {
        None => (),
        Some(en$0 @ Enum::A) => (),
    }
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn bind_pat_and_path_ignore_ref() {
        check(
            r#"
enum Enum { A, B }
fn quux(x: Option<Enum>) {
    match x {
        None => (),
        Some(ref en$0) => (),
    }
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn bind_pat_and_path() {
        check(
            r#"
enum Enum { A, B }
fn quux(x: Option<Enum>) {
    match x {
        None => (),
        Some(En$0) => (),
    }
}
"#,
            expect![[r#"
                en Enum
            "#]],
        );
    }

    #[test]
    fn completes_bindings_from_let() {
        check(
            r#"
fn quux(x: i32) {
    let y = 92;
    1 + $0;
    let z = ();
}
"#,
            expect![[r#"
                lc y       i32
                lc x       i32
                fn quux(…) fn(i32)
            "#]],
        );
    }

    #[test]
    fn completes_bindings_from_if_let() {
        check(
            r#"
fn quux() {
    if let Some(x) = foo() {
        let y = 92;
    };
    if let Some(a) = bar() {
        let b = 62;
        1 + $0
    }
}
"#,
            expect![[r#"
                lc b      i32
                lc a
                fn quux() fn()
            "#]],
        );
    }

    #[test]
    fn completes_bindings_from_for() {
        check(
            r#"
fn quux() {
    for x in &[1, 2, 3] { $0 }
}
"#,
            expect![[r#"
                lc x
                fn quux() fn()
            "#]],
        );
    }

    #[test]
    fn completes_if_prefix_is_keyword() {
        cov_mark::check!(completes_if_prefix_is_keyword);
        check_edit(
            "wherewolf",
            r#"
fn main() {
    let wherewolf = 92;
    drop(where$0)
}
"#,
            r#"
fn main() {
    let wherewolf = 92;
    drop(wherewolf)
}
"#,
        )
    }

    #[test]
    fn completes_generic_params() {
        check(
            r#"fn quux<T>() { $0 }"#,
            expect![[r#"
                tp T
                fn quux() fn()
            "#]],
        );
        check(
            r#"fn quux<const C: usize>() { $0 }"#,
            expect![[r#"
                cp C
                fn quux() fn()
            "#]],
        );
    }

    #[test]
    fn does_not_complete_lifetimes() {
        cov_mark::check!(skip_lifetime_completion);
        check(
            r#"fn quux<'a>() { $0 }"#,
            expect![[r#"
                fn quux() fn()
            "#]],
        );
    }

    #[test]
    fn completes_generic_params_in_struct() {
        check(
            r#"struct S<T> { x: $0}"#,
            expect![[r#"
                sp Self
                tp T
                st S<…>
            "#]],
        );
    }

    #[test]
    fn completes_self_in_enum() {
        check(
            r#"enum X { Y($0) }"#,
            expect![[r#"
                sp Self
                en X
            "#]],
        );
    }

    #[test]
    fn completes_module_items() {
        check(
            r#"
struct S;
enum E {}
fn quux() { $0 }
"#,
            expect![[r#"
                st S
                fn quux() fn()
                en E
            "#]],
        );
    }

    /// Regression test for issue #6091.
    #[test]
    fn correctly_completes_module_items_prefixed_with_underscore() {
        check_edit(
            "_alpha",
            r#"
fn main() {
    _$0
}
fn _alpha() {}
"#,
            r#"
fn main() {
    _alpha()$0
}
fn _alpha() {}
"#,
        )
    }

    #[test]
    fn completes_extern_prelude() {
        check(
            r#"
//- /lib.rs crate:main deps:other_crate
use $0;

//- /other_crate/lib.rs crate:other_crate
// nothing here
"#,
            expect![[r#"
                md other_crate
            "#]],
        );
    }

    #[test]
    fn completes_module_items_in_nested_modules() {
        check(
            r#"
struct Foo;
mod m {
    struct Bar;
    fn quux() { $0 }
}
"#,
            expect![[r#"
                fn quux() fn()
                st Bar
            "#]],
        );
    }

    #[test]
    fn completes_return_type() {
        check(
            r#"
struct Foo;
fn x() -> $0
"#,
            expect![[r#"
                st Foo
                fn x() fn()
            "#]],
        );
    }

    #[test]
    fn dont_show_both_completions_for_shadowing() {
        check(
            r#"
fn foo() {
    let bar = 92;
    {
        let bar = 62;
        drop($0)
    }
}
"#,
            // FIXME: should be only one bar here
            expect![[r#"
                lc bar   i32
                lc bar   i32
                fn foo() fn()
            "#]],
        );
    }

    #[test]
    fn completes_self_in_methods() {
        check(
            r#"impl S { fn foo(&self) { $0 } }"#,
            expect![[r#"
                lc self &{unknown}
                sp Self
            "#]],
        );
    }

    #[test]
    fn completes_prelude() {
        check(
            r#"
//- /main.rs crate:main deps:std
fn foo() { let x: $0 }

//- /std/lib.rs crate:std
#[prelude_import]
use prelude::*;

mod prelude { struct Option; }
"#,
            expect![[r#"
                fn foo()  fn()
                md std
                st Option
            "#]],
        );
    }

    #[test]
    fn completes_prelude_macros() {
        check(
            r#"
//- /main.rs crate:main deps:std
fn f() {$0}

//- /std/lib.rs crate:std
#[prelude_import]
pub use prelude::*;

#[macro_use]
mod prelude {
    pub use crate::concat;
}

mod macros {
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat { }
}
"#,
            expect![[r##"
                fn f()        fn()
                ma concat!(…) #[macro_export] macro_rules! concat
                md std
            "##]],
        );
    }

    #[test]
    fn completes_std_prelude_if_core_is_defined() {
        check(
            r#"
//- /main.rs crate:main deps:core,std
fn foo() { let x: $0 }

//- /core/lib.rs crate:core
#[prelude_import]
use prelude::*;

mod prelude { struct Option; }

//- /std/lib.rs crate:std deps:core
#[prelude_import]
use prelude::*;

mod prelude { struct String; }
"#,
            expect![[r#"
                fn foo()  fn()
                md std
                md core
                st String
            "#]],
        );
    }

    #[test]
    fn completes_macros_as_value() {
        check(
            r#"
macro_rules! foo { () => {} }

#[macro_use]
mod m1 {
    macro_rules! bar { () => {} }
}

mod m2 {
    macro_rules! nope { () => {} }

    #[macro_export]
    macro_rules! baz { () => {} }
}

fn main() { let v = $0 }
"#,
            expect![[r##"
                md m1
                ma baz!(…) #[macro_export] macro_rules! baz
                fn main()  fn()
                md m2
                ma bar!(…) macro_rules! bar
                ma foo!(…) macro_rules! foo
            "##]],
        );
    }

    #[test]
    fn completes_both_macro_and_value() {
        check(
            r#"
macro_rules! foo { () => {} }
fn foo() { $0 }
"#,
            expect![[r#"
                fn foo()   fn()
                ma foo!(…) macro_rules! foo
            "#]],
        );
    }

    #[test]
    fn completes_macros_as_type() {
        check(
            r#"
macro_rules! foo { () => {} }
fn main() { let x: $0 }
"#,
            expect![[r#"
                fn main()  fn()
                ma foo!(…) macro_rules! foo
            "#]],
        );
    }

    #[test]
    fn completes_macros_as_stmt() {
        check(
            r#"
macro_rules! foo { () => {} }
fn main() { $0 }
"#,
            expect![[r#"
                fn main()  fn()
                ma foo!(…) macro_rules! foo
            "#]],
        );
    }

    #[test]
    fn completes_local_item() {
        check(
            r#"
fn main() {
    return f$0;
    fn frobnicate() {}
}
"#,
            expect![[r#"
                fn frobnicate() fn()
                fn main()       fn()
            "#]],
        );
    }

    #[test]
    fn completes_in_simple_macro_1() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!($0);
}
"#,
            expect![[r#"
                lc y       i32
                lc x       i32
                fn quux(…) fn(i32)
                ma m!(…)   macro_rules! m
            "#]],
        );
    }

    #[test]
    fn completes_in_simple_macro_2() {
        check(
            r"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!(x$0);
}
",
            expect![[r#"
                lc y       i32
                lc x       i32
                fn quux(…) fn(i32)
                ma m!(…)   macro_rules! m
            "#]],
        );
    }

    #[test]
    fn completes_in_simple_macro_without_closing_parens() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn quux(x: i32) {
    let y = 92;
    m!(x$0
}
"#,
            expect![[r#"
                lc y       i32
                lc x       i32
                fn quux(…) fn(i32)
                ma m!(…)   macro_rules! m
            "#]],
        );
    }

    #[test]
    fn completes_unresolved_uses() {
        check(
            r#"
use spam::Quux;

fn main() { $0 }
"#,
            expect![[r#"
                fn main() fn()
                ?? Quux
            "#]],
        );
    }

    #[test]
    fn completes_enum_variant_basic_expr() {
        check(
            r#"
enum Foo { Bar, Baz, Quux }
fn main() { let foo: Foo = Q$0 }
"#,
            expect![[r#"
                ev Foo::Bar  ()
                ev Foo::Baz  ()
                ev Foo::Quux ()
                en Foo
                fn main()    fn()
            "#]],
        )
    }

    #[test]
    fn completes_enum_variant_from_module() {
        check(
            r#"
mod m { pub enum E { V } }
fn f() -> m::E { V$0 }
"#,
            expect![[r#"
                ev m::E::V ()
                md m
                fn f()     fn() -> E
            "#]],
        )
    }

    #[test]
    fn dont_complete_attr() {
        check(
            r#"
struct Foo;
#[$0]
fn f() {}
"#,
            expect![[""]],
        )
    }

    #[test]
    fn completes_type_or_trait_in_impl_block() {
        check(
            r#"
trait MyTrait {}
struct MyStruct {}

impl My$0
"#,
            expect![[r#"
                sp Self
                tt MyTrait
                st MyStruct
            "#]],
        )
    }
}
