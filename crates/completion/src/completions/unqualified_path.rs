//! Completion of names from the current scope, e.g. locals and imported items.

use either::Either;
use hir::{Adt, ModuleDef, ScopeDef, Type};
use ide_db::helpers::insert_use::ImportScope;
use ide_db::imports_locator;
use syntax::AstNode;
use test_utils::mark;

use crate::{
    render::{render_resolution_with_import, RenderContext},
    CompletionContext, Completions, ImportEdit,
};

pub(crate) fn complete_unqualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path || ctx.is_pat_binding_or_const) {
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
        complete_enum_variants(acc, ctx, ty);
    }

    if ctx.is_pat_binding_or_const {
        return;
    }

    ctx.scope.process_all_names(&mut |name, res| {
        if ctx.use_item_syntax.is_some() {
            if let (ScopeDef::Unknown, Some(name_ref)) = (&res, &ctx.name_ref_syntax) {
                if name_ref.syntax().text() == name.to_string().as_str() {
                    mark::hit!(self_fulfilling_completion);
                    return;
                }
            }
        }
        acc.add_resolution(ctx, name.to_string(), &res)
    });

    if ctx.config.enable_autoimport_completions && ctx.config.resolve_additional_edits_lazily() {
        fuzzy_completion(acc, ctx);
    }
}

fn complete_enum_variants(acc: &mut Completions, ctx: &CompletionContext, ty: &Type) {
    if let Some(Adt::Enum(enum_data)) = ty.as_adt() {
        let variants = enum_data.variants(ctx.db);

        let module = if let Some(module) = ctx.scope.module() {
            // Compute path from the completion site if available.
            module
        } else {
            // Otherwise fall back to the enum's definition site.
            enum_data.module(ctx.db)
        };

        for variant in variants {
            if let Some(path) = module.find_use_path(ctx.db, ModuleDef::from(variant)) {
                // Variants with trivial paths are already added by the existing completion logic,
                // so we should avoid adding these twice
                if path.segments.len() > 1 {
                    acc.add_qualified_enum_variant(ctx, variant, path);
                }
            }
        }
    }
}

// Feature: Fuzzy Completion and Autoimports
//
// When completing names in the current scope, proposes additional imports from other modules or crates,
// if they can be qualified in the scope and their name contains all symbols from the completion input
// (case-insensitive, in any order or places).
//
// ```
// fn main() {
//     pda<|>
// }
// # pub mod std { pub mod marker { pub struct PhantomData { } } }
// ```
// ->
// ```
// use std::marker::PhantomData;
//
// fn main() {
//     PhantomData
// }
// # pub mod std { pub mod marker { pub struct PhantomData { } } }
// ```
//
// .Fuzzy search details
//
// To avoid an excessive amount of the results returned, completion input is checked for inclusion in the identifiers only
// (i.e. in `HashMap` in the `std::collections::HashMap` path), also not in the module indentifiers.
//
// .Merge Behavior
//
// It is possible to configure how use-trees are merged with the `importMergeBehavior` setting.
// Mimics the corresponding behavior of the `Auto Import` feature.
//
// .LSP and performance implications
//
// The feature is enabled only if the LSP client supports LSP protocol version 3.16+ and reports the `additionalTextEdits`
// (case sensitive) resolve client capability in its client capabilities.
// This way the server is able to defer the costly computations, doing them for a selected completion item only.
// For clients with no such support, all edits have to be calculated on the completion request, including the fuzzy search completion ones,
// which might be slow ergo the feature is automatically disabled.
//
// .Feature toggle
//
// The feature can be forcefully turned off in the settings with the `rust-analyzer.completion.enableAutoimportCompletions` flag.
// Note that having this flag set to `true` does not guarantee that the feature is enabled: your client needs to have the corredponding
// capability enabled.
fn fuzzy_completion(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let _p = profile::span("fuzzy_completion");
    let potential_import_name = ctx.token.to_string();

    let current_module = ctx.scope.module()?;
    let anchor = ctx.name_ref_syntax.as_ref()?;
    let import_scope = ImportScope::find_insert_use_container(anchor.syntax(), &ctx.sema)?;

    let possible_imports = imports_locator::find_similar_imports(
        &ctx.sema,
        ctx.krate?,
        Some(100),
        &potential_import_name,
        true,
    )
    .filter_map(|import_candidate| {
        Some(match import_candidate {
            Either::Left(module_def) => {
                (current_module.find_use_path(ctx.db, module_def)?, ScopeDef::ModuleDef(module_def))
            }
            Either::Right(macro_def) => {
                (current_module.find_use_path(ctx.db, macro_def)?, ScopeDef::MacroDef(macro_def))
            }
        })
    })
    .filter(|(mod_path, _)| mod_path.len() > 1)
    .filter_map(|(import_path, definition)| {
        render_resolution_with_import(
            RenderContext::new(ctx),
            ImportEdit {
                import_path: import_path.clone(),
                import_scope: import_scope.clone(),
                merge_behavior: ctx.config.merge,
            },
            &definition,
        )
    });

    acc.add_all(possible_imports);
    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::{
        test_utils::{check_edit, check_edit_with_config, completion_list},
        CompletionConfig, CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
    }

    #[test]
    fn self_fulfilling_completion() {
        mark::check!(self_fulfilling_completion);
        check(
            r#"
use foo<|>
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
        Some(en<|> @ Enum::A) => (),
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
        Some(ref en<|>) => (),
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
        Some(En<|>) => (),
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
    1 + <|>;
    let z = ();
}
"#,
            expect![[r#"
                fn quux(…) fn quux(x: i32)
                bn x       i32
                bn y       i32
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
        1 + <|>
    }
}
"#,
            expect![[r#"
                bn a
                bn b      i32
                fn quux() fn quux()
            "#]],
        );
    }

    #[test]
    fn completes_bindings_from_for() {
        check(
            r#"
fn quux() {
    for x in &[1, 2, 3] { <|> }
}
"#,
            expect![[r#"
                fn quux() fn quux()
                bn x
            "#]],
        );
    }

    #[test]
    fn completes_if_prefix_is_keyword() {
        mark::check!(completes_if_prefix_is_keyword);
        check_edit(
            "wherewolf",
            r#"
fn main() {
    let wherewolf = 92;
    drop(where<|>)
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
            r#"fn quux<T>() { <|> }"#,
            expect![[r#"
                tp T
                fn quux() fn quux<T>()
            "#]],
        );
    }

    #[test]
    fn completes_generic_params_in_struct() {
        check(
            r#"struct S<T> { x: <|>}"#,
            expect![[r#"
                st S<…>
                tp Self
                tp T
            "#]],
        );
    }

    #[test]
    fn completes_self_in_enum() {
        check(
            r#"enum X { Y(<|>) }"#,
            expect![[r#"
                tp Self
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
fn quux() { <|> }
"#,
            expect![[r#"
                en E
                st S
                fn quux() fn quux()
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
    _<|>
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
use <|>;

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
    fn quux() { <|> }
}
"#,
            expect![[r#"
                st Bar
                fn quux() fn quux()
            "#]],
        );
    }

    #[test]
    fn completes_return_type() {
        check(
            r#"
struct Foo;
fn x() -> <|>
"#,
            expect![[r#"
                st Foo
                fn x() fn x()
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
        drop(<|>)
    }
}
"#,
            // FIXME: should be only one bar here
            expect![[r#"
                bn bar   i32
                bn bar   i32
                fn foo() fn foo()
            "#]],
        );
    }

    #[test]
    fn completes_self_in_methods() {
        check(
            r#"impl S { fn foo(&self) { <|> } }"#,
            expect![[r#"
                tp Self
                bn self &{unknown}
            "#]],
        );
    }

    #[test]
    fn completes_prelude() {
        check(
            r#"
//- /main.rs crate:main deps:std
fn foo() { let x: <|> }

//- /std/lib.rs crate:std
#[prelude_import]
use prelude::*;

mod prelude { struct Option; }
"#,
            expect![[r#"
                st Option
                fn foo()  fn foo()
                md std
            "#]],
        );
    }

    #[test]
    fn completes_std_prelude_if_core_is_defined() {
        check(
            r#"
//- /main.rs crate:main deps:core,std
fn foo() { let x: <|> }

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
                st String
                md core
                fn foo()  fn foo()
                md std
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

fn main() { let v = <|> }
"#,
            expect![[r##"
                ma bar!(…) macro_rules! bar
                ma baz!(…) #[macro_export]
                macro_rules! baz
                ma foo!(…) macro_rules! foo
                md m1
                md m2
                fn main()  fn main()
            "##]],
        );
    }

    #[test]
    fn completes_both_macro_and_value() {
        check(
            r#"
macro_rules! foo { () => {} }
fn foo() { <|> }
"#,
            expect![[r#"
                ma foo!(…) macro_rules! foo
                fn foo()   fn foo()
            "#]],
        );
    }

    #[test]
    fn completes_macros_as_type() {
        check(
            r#"
macro_rules! foo { () => {} }
fn main() { let x: <|> }
"#,
            expect![[r#"
                ma foo!(…) macro_rules! foo
                fn main()  fn main()
            "#]],
        );
    }

    #[test]
    fn completes_macros_as_stmt() {
        check(
            r#"
macro_rules! foo { () => {} }
fn main() { <|> }
"#,
            expect![[r#"
                ma foo!(…) macro_rules! foo
                fn main()  fn main()
            "#]],
        );
    }

    #[test]
    fn completes_local_item() {
        check(
            r#"
fn main() {
    return f<|>;
    fn frobnicate() {}
}
"#,
            expect![[r#"
                fn frobnicate() fn frobnicate()
                fn main()       fn main()
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
    m!(<|>);
}
"#,
            expect![[r#"
                ma m!(…)   macro_rules! m
                fn quux(…) fn quux(x: i32)
                bn x       i32
                bn y       i32
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
    m!(x<|>);
}
",
            expect![[r#"
                ma m!(…)   macro_rules! m
                fn quux(…) fn quux(x: i32)
                bn x       i32
                bn y       i32
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
    m!(x<|>
}
"#,
            expect![[r#"
                ma m!(…)   macro_rules! m
                fn quux(…) fn quux(x: i32)
                bn x       i32
                bn y       i32
            "#]],
        );
    }

    #[test]
    fn completes_unresolved_uses() {
        check(
            r#"
use spam::Quux;

fn main() { <|> }
"#,
            expect![[r#"
                ?? Quux
                fn main() fn main()
            "#]],
        );
    }
    #[test]
    fn completes_enum_variant_matcharm() {
        check(
            r#"
enum Foo { Bar, Baz, Quux }

fn main() {
    let foo = Foo::Quux;
    match foo { Qu<|> }
}
"#,
            expect![[r#"
                en Foo
                ev Foo::Bar  ()
                ev Foo::Baz  ()
                ev Foo::Quux ()
            "#]],
        )
    }

    #[test]
    fn completes_enum_variant_iflet() {
        check(
            r#"
enum Foo { Bar, Baz, Quux }

fn main() {
    let foo = Foo::Quux;
    if let Qu<|> = foo { }
}
"#,
            expect![[r#"
                en Foo
                ev Foo::Bar  ()
                ev Foo::Baz  ()
                ev Foo::Quux ()
            "#]],
        )
    }

    #[test]
    fn completes_enum_variant_basic_expr() {
        check(
            r#"
enum Foo { Bar, Baz, Quux }
fn main() { let foo: Foo = Q<|> }
"#,
            expect![[r#"
                en Foo
                ev Foo::Bar  ()
                ev Foo::Baz  ()
                ev Foo::Quux ()
                fn main()    fn main()
            "#]],
        )
    }

    #[test]
    fn completes_enum_variant_from_module() {
        check(
            r#"
mod m { pub enum E { V } }
fn f() -> m::E { V<|> }
"#,
            expect![[r#"
                fn f()     fn f() -> m::E
                md m
                ev m::E::V ()
            "#]],
        )
    }

    #[test]
    fn dont_complete_attr() {
        check(
            r#"
struct Foo;
#[<|>]
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

impl My<|>
"#,
            expect![[r#"
                st MyStruct
                tt MyTrait
                tp Self
            "#]],
        )
    }

    #[test]
    fn function_fuzzy_completion() {
        let mut completion_config = CompletionConfig::default();
        completion_config
            .active_resolve_capabilities
            .insert(crate::CompletionResolveCapability::AdditionalTextEdits);

        check_edit_with_config(
            completion_config,
            "stdin",
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi<|>
}
"#,
            r#"
use dep::io::stdin;

fn main() {
    stdin()$0
}
"#,
        );
    }

    #[test]
    fn macro_fuzzy_completion() {
        let mut completion_config = CompletionConfig::default();
        completion_config
            .active_resolve_capabilities
            .insert(crate::CompletionResolveCapability::AdditionalTextEdits);

        check_edit_with_config(
            completion_config,
            "macro_with_curlies!",
            r#"
//- /lib.rs crate:dep
/// Please call me as macro_with_curlies! {}
#[macro_export]
macro_rules! macro_with_curlies {
    () => {}
}

//- /main.rs crate:main deps:dep
fn main() {
    curli<|>
}
"#,
            r#"
use dep::macro_with_curlies;

fn main() {
    macro_with_curlies! {$0}
}
"#,
        );
    }

    #[test]
    fn struct_fuzzy_completion() {
        let mut completion_config = CompletionConfig::default();
        completion_config
            .active_resolve_capabilities
            .insert(crate::CompletionResolveCapability::AdditionalTextEdits);

        check_edit_with_config(
            completion_config,
            "ThirdStruct",
            r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    pub struct SecondStruct;
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    this<|>
}
"#,
            r#"
use dep::{FirstStruct, some_module::{SecondStruct, ThirdStruct}};

fn main() {
    ThirdStruct
}
"#,
        );
    }
}
