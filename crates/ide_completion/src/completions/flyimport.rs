//! Feature: completion with imports-on-the-fly
//!
//! When completing names in the current scope, proposes additional imports from other modules or crates,
//! if they can be qualified in the scope and their name contains all symbols from the completion input
//! (case-insensitive, in any order or places).
//!
//! ```
//! fn main() {
//!     pda$0
//! }
//! # pub mod std { pub mod marker { pub struct PhantomData { } } }
//! ```
//! ->
//! ```
//! use std::marker::PhantomData;
//!
//! fn main() {
//!     PhantomData
//! }
//! # pub mod std { pub mod marker { pub struct PhantomData { } } }
//! ```
//!
//! Also completes associated items, that require trait imports.
//!
//! .Fuzzy search details
//!
//! To avoid an excessive amount of the results returned, completion input is checked for inclusion in the names only
//! (i.e. in `HashMap` in the `std::collections::HashMap` path).
//! For the same reasons, avoids searching for any path imports for inputs with their length less that 2 symbols
//! (but shows all associated items for any input length).
//!
//! .Import configuration
//!
//! It is possible to configure how use-trees are merged with the `importMergeBehavior` setting.
//! Mimics the corresponding behavior of the `Auto Import` feature.
//!
//! .LSP and performance implications
//!
//! The feature is enabled only if the LSP client supports LSP protocol version 3.16+ and reports the `additionalTextEdits`
//! (case sensitive) resolve client capability in its client capabilities.
//! This way the server is able to defer the costly computations, doing them for a selected completion item only.
//! For clients with no such support, all edits have to be calculated on the completion request, including the fuzzy search completion ones,
//! which might be slow ergo the feature is automatically disabled.
//!
//! .Feature toggle
//!
//! The feature can be forcefully turned off in the settings with the `rust-analyzer.completion.enableAutoimportCompletions` flag.
//! Note that having this flag set to `true` does not guarantee that the feature is enabled: your client needs to have the corredponding
//! capability enabled.

use hir::{AsAssocItem, ModPath, ScopeDef};
use ide_db::helpers::{
    import_assets::{ImportAssets, ImportCandidate},
    insert_use::ImportScope,
};
use rustc_hash::FxHashSet;
use syntax::{AstNode, SyntaxNode, T};

use crate::{
    context::CompletionContext,
    render::{render_resolution_with_import, RenderContext},
    ImportEdit,
};

use super::Completions;

pub(crate) fn import_on_the_fly(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !ctx.config.enable_imports_on_the_fly {
        return None;
    }
    if ctx.use_item_syntax.is_some()
        || ctx.attribute_under_caret.is_some()
        || ctx.mod_declaration_under_caret.is_some()
    {
        return None;
    }
    let potential_import_name = {
        let token_kind = ctx.token.kind();
        if matches!(token_kind, T![.] | T![::]) {
            String::new()
        } else {
            ctx.token.to_string()
        }
    };

    let _p = profile::span("import_on_the_fly").detail(|| potential_import_name.to_string());

    let user_input_lowercased = potential_import_name.to_lowercase();
    let import_assets = import_assets(ctx, potential_import_name)?;
    let import_scope = ImportScope::find_insert_use_container(
        position_for_import(ctx, Some(import_assets.import_candidate()))?,
        &ctx.sema,
    )?;

    let scope_definitions = scope_definitions(ctx);
    let mut all_mod_paths = import_assets
        .search_for_imports(&ctx.sema, ctx.config.insert_use.prefix_kind)
        .into_iter()
        .map(|import| {
            let proposed_def = match import.item_to_display() {
                hir::ItemInNs::Types(id) => ScopeDef::ModuleDef(id.into()),
                hir::ItemInNs::Values(id) => ScopeDef::ModuleDef(id.into()),
                hir::ItemInNs::Macros(id) => ScopeDef::MacroDef(id.into()),
            };
            (import, proposed_def)
        })
        .filter(|(_, proposed_def)| !scope_definitions.contains(proposed_def))
        .collect::<Vec<_>>();
    all_mod_paths.sort_by_cached_key(|(import, _)| {
        compute_fuzzy_completion_order_key(import.display_path(), &user_input_lowercased)
    });

    acc.add_all(all_mod_paths.into_iter().filter_map(|(import, definition)| {
        let import_for_trait_assoc_item = match definition {
            ScopeDef::ModuleDef(module_def) => module_def
                .as_assoc_item(ctx.db)
                .and_then(|assoc| assoc.containing_trait(ctx.db))
                .is_some(),
            _ => false,
        };
        let import_edit =
            ImportEdit { import, import_scope: import_scope.clone(), import_for_trait_assoc_item };
        render_resolution_with_import(RenderContext::new(ctx), import_edit, &definition)
    }));
    Some(())
}

fn scope_definitions(ctx: &CompletionContext) -> FxHashSet<ScopeDef> {
    let mut scope_definitions = FxHashSet::default();
    ctx.scope.process_all_names(&mut |_, scope_def| {
        scope_definitions.insert(scope_def);
    });
    scope_definitions
}

pub(crate) fn position_for_import<'a>(
    ctx: &'a CompletionContext,
    import_candidate: Option<&ImportCandidate>,
) -> Option<&'a SyntaxNode> {
    Some(match import_candidate {
        Some(ImportCandidate::Path(_)) => ctx.name_ref_syntax.as_ref()?.syntax(),
        Some(ImportCandidate::TraitAssocItem(_)) => ctx.path_qual.as_ref()?.syntax(),
        Some(ImportCandidate::TraitMethod(_)) => ctx.dot_receiver.as_ref()?.syntax(),
        None => ctx
            .name_ref_syntax
            .as_ref()
            .map(|name_ref| name_ref.syntax())
            .or_else(|| ctx.path_qual.as_ref().map(|path| path.syntax()))
            .or_else(|| ctx.dot_receiver.as_ref().map(|expr| expr.syntax()))?,
    })
}

fn import_assets(ctx: &CompletionContext, fuzzy_name: String) -> Option<ImportAssets> {
    let current_module = ctx.scope.module()?;
    if let Some(dot_receiver) = &ctx.dot_receiver {
        ImportAssets::for_fuzzy_method_call(
            current_module,
            ctx.sema.type_of_expr(dot_receiver)?,
            fuzzy_name,
        )
    } else {
        let fuzzy_name_length = fuzzy_name.len();
        let assets_for_path = ImportAssets::for_fuzzy_path(
            current_module,
            ctx.path_qual.clone(),
            fuzzy_name,
            &ctx.sema,
        )?;

        if matches!(assets_for_path.import_candidate(), ImportCandidate::Path(_))
            && fuzzy_name_length < 2
        {
            cov_mark::hit!(ignore_short_input_for_path);
            None
        } else {
            Some(assets_for_path)
        }
    }
}

fn compute_fuzzy_completion_order_key(
    proposed_mod_path: &ModPath,
    user_input_lowercased: &str,
) -> usize {
    cov_mark::hit!(certain_fuzzy_order_test);
    let import_name = match proposed_mod_path.segments().last() {
        Some(name) => name.to_string().to_lowercase(),
        None => return usize::MAX,
    };
    match import_name.match_indices(user_input_lowercased).next() {
        Some((first_matching_index, _)) => first_matching_index,
        None => usize::MAX,
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        item::CompletionKind,
        test_utils::{check_edit, check_edit_with_config, completion_list, TEST_CONFIG},
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual);
    }

    #[test]
    fn function_fuzzy_completion() {
        check_edit(
            "stdin",
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi$0
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
        check_edit(
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
    curli$0
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
        check_edit(
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
    this$0
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

    #[test]
    fn short_paths_are_ignored() {
        cov_mark::check!(ignore_short_input_for_path);

        check(
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
    t$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn fuzzy_completions_come_in_specific_order() {
        cov_mark::check!(certain_fuzzy_order_test);
        check(
            r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    // already imported, omitted
    pub struct SecondStruct;
    // does not contain all letters from the query, omitted
    pub struct UnrelatedOne;
    // contains all letters from the query, but not in sequence, displayed last
    pub struct ThiiiiiirdStruct;
    // contains all letters from the query, but not in the beginning, displayed second
    pub struct AfterThirdStruct;
    // contains all letters from the query in the begginning, displayed first
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    hir$0
}
"#,
            expect![[r#"
                st dep::some_module::ThirdStruct
                st dep::some_module::AfterThirdStruct
                st dep::some_module::ThiiiiiirdStruct
            "#]],
        );
    }

    #[test]
    fn trait_function_fuzzy_completion() {
        let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            dep::test_mod::TestStruct::wei$0
        }
        "#;

        check(
            fixture,
            expect![[r#"
                fn weird_function() (dep::test_mod::TestTrait) -> ()
            "#]],
        );

        check_edit(
            "weird_function",
            fixture,
            r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::weird_function()$0
}
"#,
        );
    }

    #[test]
    fn trait_const_fuzzy_completion() {
        let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            dep::test_mod::TestStruct::spe$0
        }
        "#;

        check(
            fixture,
            expect![[r#"
            ct SPECIAL_CONST (dep::test_mod::TestTrait)
        "#]],
        );

        check_edit(
            "SPECIAL_CONST",
            fixture,
            r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::SPECIAL_CONST
}
"#,
        );
    }

    #[test]
    fn trait_method_fuzzy_completion() {
        let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            let test_struct = dep::test_mod::TestStruct {};
            test_struct.ran$0
        }
        "#;

        check(
            fixture,
            expect![[r#"
                me random_method() (dep::test_mod::TestTrait) -> ()
            "#]],
        );

        check_edit(
            "random_method",
            fixture,
            r#"
use dep::test_mod::TestTrait;

fn main() {
    let test_struct = dep::test_mod::TestStruct {};
    test_struct.random_method()$0
}
"#,
        );
    }

    #[test]
    fn no_trait_type_fuzzy_completion() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::hum$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn does_not_propose_names_in_scope() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
use dep::test_mod::TestStruct;
fn main() {
    TestSt$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn does_not_propose_traits_in_scope() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
use dep::test_mod::{TestStruct, TestTrait};
fn main() {
    dep::test_mod::TestStruct::hum$0
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn blanket_trait_impl_import() {
        check_edit(
            "another_function",
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub struct TestStruct {}
    pub trait TestTrait {
        fn another_function();
    }
    impl<T> TestTrait for T {
        fn another_function() {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::ano$0
}
"#,
            r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::another_function()$0
}
"#,
        );
    }

    #[test]
    fn zero_input_deprecated_assoc_item_completion() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    #[deprecated]
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    let test_struct = dep::test_mod::TestStruct {};
    test_struct.$0
}
        "#,
            expect![[r#"
                me random_method() (dep::test_mod::TestTrait) -> () DEPRECATED
            "#]],
        );

        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    #[deprecated]
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::$0
}
"#,
            expect![[r#"
                fn weird_function() (dep::test_mod::TestTrait) -> () DEPRECATED
                ct SPECIAL_CONST (dep::test_mod::TestTrait) DEPRECATED
            "#]],
        );
    }

    #[test]
    fn no_completions_in_use_statements() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
use stdi$0

fn main() {}
"#,
            expect![[]],
        );
    }

    #[test]
    fn prefix_config_usage() {
        let fixture = r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar;

fn main() {
    Ite$0
}"#;
        let mut config = TEST_CONFIG;

        config.insert_use.prefix_kind = hir::PrefixKind::ByCrate;
        check_edit_with_config(
            config.clone(),
            "Item",
            fixture,
            r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar::{self, Item};

fn main() {
    Item
}"#,
        );

        config.insert_use.prefix_kind = hir::PrefixKind::BySelf;
        check_edit_with_config(
            config.clone(),
            "Item",
            fixture,
            r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar;

use self::foo::bar::Item;

fn main() {
    Item
}"#,
        );

        config.insert_use.prefix_kind = hir::PrefixKind::Plain;
        check_edit_with_config(
            config,
            "Item",
            fixture,
            r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use foo::bar::Item;

use crate::foo::bar;

fn main() {
    Item
}"#,
        );
    }

    #[test]
    fn unresolved_qualifier() {
        check_edit(
            "Item",
            r#"
mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Item;
        }
    }
}

fn main() {
    bar::baz::Ite$0
}
"#,
            r#"
use foo::bar;

mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Item;
        }
    }
}

fn main() {
    bar::baz::Item
}
"#,
        );
    }

    #[test]
    fn unresolved_assoc_item_container() {
        check_edit(
            "TEST_ASSOC",
            r#"
mod foo {
    pub struct Item;

    impl Item {
        pub const TEST_ASSOC: usize = 3;
    }
}

fn main() {
    Item::TEST_A$0
}
"#,
            r#"
use foo::Item;

mod foo {
    pub struct Item;

    impl Item {
        pub const TEST_ASSOC: usize = 3;
    }
}

fn main() {
    Item::TEST_ASSOC
}
"#,
        );
    }

    #[test]
    fn unresolved_assoc_item_container_with_path() {
        check_edit(
            "TEST_ASSOC",
            r#"
mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    bar::Item::TEST_A$0
}
"#,
            r#"
use foo::bar;

mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    bar::Item::TEST_ASSOC
}
"#,
        );
    }

    #[test]
    fn unresolved_assoc_item_container_and_trait_with_path() {
        check_edit(
            "TEST_ASSOC",
            r#"
mod foo {
    pub mod bar {
        pub trait SomeTrait {
            const TEST_ASSOC: usize;
        }
    }

    pub mod baz {
        use super::bar::SomeTrait;

        pub struct Item;

        impl SomeTrait for Item {
            const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    baz::Item::TEST_A$0
}
"#,
            r#"
use foo::{bar::SomeTrait, baz};

mod foo {
    pub mod bar {
        pub trait SomeTrait {
            const TEST_ASSOC: usize;
        }
    }

    pub mod baz {
        use super::bar::SomeTrait;

        pub struct Item;

        impl SomeTrait for Item {
            const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    baz::Item::TEST_ASSOC
}
"#,
        );
    }
}
