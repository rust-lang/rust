//! Completion of names from the current scope, e.g. locals and imported items.

use hir::ScopeDef;
use syntax::{ast, AstNode};

use crate::{patterns::ImmediateLocation, CompletionContext, Completions};

pub(crate) fn complete_unqualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.is_path_disallowed() || !ctx.is_trivial_path() || ctx.has_impl_or_trait_prev_sibling() {
        return;
    }

    if ctx.in_use_tree() {
        // only show modules in a fresh UseTree
        cov_mark::hit!(only_completes_modules_in_import);
        ctx.scope.process_all_names(&mut |name, res| {
            if let ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = res {
                acc.add_resolution(ctx, name, &res);
            }
        });

        std::array::IntoIter::new(["self::", "super::", "crate::"])
            .for_each(|kw| acc.add_keyword(ctx, kw));
        return;
    }
    std::array::IntoIter::new(["self", "super", "crate"]).for_each(|kw| acc.add_keyword(ctx, kw));

    if ctx.expects_item() || ctx.expects_assoc_item() {
        // only show macros in {Assoc}ItemList
        ctx.scope.process_all_names(&mut |name, res| {
            if let hir::ScopeDef::MacroDef(mac) = res {
                if mac.is_fn_like() {
                    acc.add_macro(ctx, Some(name.clone()), mac);
                }
            }
            if let hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = res {
                acc.add_resolution(ctx, name, &res);
            }
        });
        return;
    }

    if matches!(&ctx.completion_location, Some(ImmediateLocation::TypeBound)) {
        ctx.scope.process_all_names(&mut |name, res| {
            let add_resolution = match res {
                ScopeDef::MacroDef(mac) => mac.is_fn_like(),
                ScopeDef::ModuleDef(hir::ModuleDef::Trait(_) | hir::ModuleDef::Module(_)) => true,
                _ => false,
            };
            if add_resolution {
                acc.add_resolution(ctx, name, &res);
            }
        });
        return;
    }

    if !ctx.expects_type() {
        if let Some(hir::Adt::Enum(e)) =
            ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
        {
            super::enum_variants_with_paths(acc, ctx, e, |acc, ctx, variant, path| {
                acc.add_qualified_enum_variant(ctx, variant, path)
            });
        }
    }

    if let Some(ImmediateLocation::GenericArgList(arg_list)) = &ctx.completion_location {
        if let Some(path_seg) = arg_list.syntax().parent().and_then(ast::PathSegment::cast) {
            if let Some(hir::PathResolution::Def(hir::ModuleDef::Trait(trait_))) =
                ctx.sema.resolve_path(&path_seg.parent_path())
            {
                trait_.items(ctx.sema.db).into_iter().for_each(|it| {
                    if let hir::AssocItem::TypeAlias(alias) = it {
                        acc.add_type_alias_with_eq(ctx, alias)
                    }
                });
            }
        }
    }

    ctx.scope.process_all_names(&mut |name, res| {
        if let ScopeDef::GenericParam(hir::GenericParam::LifetimeParam(_)) | ScopeDef::Label(_) =
            res
        {
            cov_mark::hit!(skip_lifetime_completion);
            return;
        }
        let add_resolution = match res {
            ScopeDef::ImplSelfType(_) => {
                !ctx.previous_token_is(syntax::T![impl]) && !ctx.previous_token_is(syntax::T![for])
            }
            // Don't suggest attribute macros and derives.
            ScopeDef::MacroDef(mac) => mac.is_fn_like(),
            // no values in type places
            ScopeDef::ModuleDef(
                hir::ModuleDef::Function(_)
                | hir::ModuleDef::Variant(_)
                | hir::ModuleDef::Static(_),
            )
            | ScopeDef::Local(_) => !ctx.expects_type(),
            // unless its a constant in a generic arg list position
            ScopeDef::ModuleDef(hir::ModuleDef::Const(_))
            | ScopeDef::GenericParam(hir::GenericParam::ConstParam(_)) => {
                !ctx.expects_type() || ctx.expects_generic_arg()
            }
            _ => true,
        };
        if add_resolution {
            acc.add_resolution(ctx, name, &res);
        }
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        tests::{check_edit, filtered_completion_list_with_config, TEST_CONFIG},
        CompletionConfig, CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        check_with_config(TEST_CONFIG, ra_fixture, expect);
    }

    fn check_with_config(config: CompletionConfig, ra_fixture: &str, expect: Expect) {
        let actual =
            filtered_completion_list_with_config(config, ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
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
pub mod prelude {
    pub mod rust_2018 {
        pub struct Option;
    }
}
"#,
            expect![[r#"
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
pub mod prelude {
    pub mod rust_2018 {
        pub use crate::concat;
    }
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
    fn does_not_complete_non_fn_macros() {
        check(
            r#"
#[rustc_builtin_macro]
pub macro Clone {}

fn f() {$0}
"#,
            expect![[r#"
                fn f() fn()
            "#]],
        );
        check(
            r#"
#[rustc_builtin_macro]
pub macro bench {}

fn f() {$0}
"#,
            expect![[r#"
                fn f() fn()
            "#]],
        );
    }

    #[test]
    fn completes_std_prelude_if_core_is_defined() {
        check(
            r#"
//- /main.rs crate:main deps:core,std
fn foo() { let x: $0 }

//- /core/lib.rs crate:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct Option;
    }
}

//- /std/lib.rs crate:std deps:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct String;
    }
}
"#,
            expect![[r#"
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
}
