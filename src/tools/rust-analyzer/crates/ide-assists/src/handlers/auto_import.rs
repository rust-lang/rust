use std::cmp::Reverse;

use either::Either;
use hir::{Module, Type, db::HirDatabase};
use ide_db::{
    active_parameter::ActiveParameter,
    helpers::mod_path_to_ast,
    imports::{
        import_assets::{ImportAssets, ImportCandidate, LocatedImport},
        insert_use::{ImportScope, insert_use, insert_use_as_alias},
    },
};
use syntax::{AstNode, Edition, SyntaxNode, ast, match_ast};

use crate::{AssistContext, AssistId, Assists, GroupLabel};

// Feature: Auto Import
//
// Using the `auto-import` assist it is possible to insert missing imports for unresolved items.
// When inserting an import it will do so in a structured manner by keeping imports grouped,
// separated by a newline in the following order:
//
// - `std` and `core`
// - External Crates
// - Current Crate, paths prefixed by `crate`
// - Current Module, paths prefixed by `self`
// - Super Module, paths prefixed by `super`
//
// Example:
// ```rust
// use std::fs::File;
//
// use itertools::Itertools;
// use syntax::ast;
//
// use crate::utils::insert_use;
//
// use self::auto_import;
//
// use super::AssistContext;
// ```
//
// #### Import Granularity
//
// It is possible to configure how use-trees are merged with the `imports.granularity.group` setting.
// It has the following configurations:
//
// - `crate`: Merge imports from the same crate into a single use statement. This kind of
//  nesting is only supported in Rust versions later than 1.24.
// - `module`: Merge imports from the same module into a single use statement.
// - `item`: Don't merge imports at all, creating one import per item.
// - `preserve`: Do not change the granularity of any imports. For auto-import this has the same
//  effect as `item`.
// - `one`: Merge all imports into a single use statement as long as they have the same visibility
//  and attributes.
//
// In `VS Code` the configuration for this is `rust-analyzer.imports.granularity.group`.
//
// #### Import Prefix
//
// The style of imports in the same crate is configurable through the `imports.prefix` setting.
// It has the following configurations:
//
// - `crate`: This setting will force paths to be always absolute, starting with the `crate`
//  prefix, unless the item is defined outside of the current crate.
// - `self`: This setting will force paths that are relative to the current module to always
//  start with `self`. This will result in paths that always start with either `crate`, `self`,
//  `super` or an extern crate identifier.
// - `plain`: This setting does not impose any restrictions in imports.
//
// In `VS Code` the configuration for this is `rust-analyzer.imports.prefix`.
//
// ![Auto Import](https://user-images.githubusercontent.com/48062697/113020673-b85be580-917a-11eb-9022-59585f35d4f8.gif)

// Assist: auto_import
//
// If the name is unresolved, provides all possible imports for it.
//
// ```
// fn main() {
//     let map = HashMap$0::new();
// }
// # pub mod std { pub mod collections { pub struct HashMap { } } }
// ```
// ->
// ```
// use std::collections::HashMap;
//
// fn main() {
//     let map = HashMap::new();
// }
// # pub mod std { pub mod collections { pub struct HashMap { } } }
// ```
pub(crate) fn auto_import(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let cfg = ctx.config.import_path_config();

    let (import_assets, syntax_under_caret, expected) = find_importable_node(ctx)?;
    let mut proposed_imports: Vec<_> = import_assets
        .search_for_imports(&ctx.sema, cfg, ctx.config.insert_use.prefix_kind)
        .collect();
    if proposed_imports.is_empty() {
        return None;
    }

    let range = ctx.sema.original_range(&syntax_under_caret).range;
    let scope = ImportScope::find_insert_use_container(&syntax_under_caret, &ctx.sema)?;

    // we aren't interested in different namespaces
    proposed_imports.sort_by(|a, b| a.import_path.cmp(&b.import_path));
    proposed_imports.dedup_by(|a, b| a.import_path == b.import_path);

    let current_module = ctx.sema.scope(scope.as_syntax_node()).map(|scope| scope.module());
    // prioritize more relevant imports
    proposed_imports.sort_by_key(|import| {
        Reverse(relevance_score(ctx, import, expected.as_ref(), current_module.as_ref()))
    });
    let edition = current_module.map(|it| it.krate().edition(ctx.db())).unwrap_or(Edition::CURRENT);

    let group_label = group_label(import_assets.import_candidate());
    for import in proposed_imports {
        let import_path = import.import_path;

        let (assist_id, import_name) =
            (AssistId::quick_fix("auto_import"), import_path.display(ctx.db(), edition));
        acc.add_group(
            &group_label,
            assist_id,
            format!("Import `{import_name}`"),
            range,
            |builder| {
                let scope = builder.make_import_scope_mut(scope.clone());
                insert_use(&scope, mod_path_to_ast(&import_path, edition), &ctx.config.insert_use);
            },
        );

        match import_assets.import_candidate() {
            ImportCandidate::TraitAssocItem(name) | ImportCandidate::TraitMethod(name) => {
                let is_method =
                    matches!(import_assets.import_candidate(), ImportCandidate::TraitMethod(_));
                let type_ = if is_method { "method" } else { "item" };
                let group_label = GroupLabel(format!(
                    "Import a trait for {} {} by alias",
                    type_,
                    name.assoc_item_name.text()
                ));
                acc.add_group(
                    &group_label,
                    assist_id,
                    format!("Import `{import_name} as _`"),
                    range,
                    |builder| {
                        let scope = builder.make_import_scope_mut(scope.clone());
                        insert_use_as_alias(
                            &scope,
                            mod_path_to_ast(&import_path, edition),
                            &ctx.config.insert_use,
                        );
                    },
                );
            }
            _ => {}
        }
    }
    Some(())
}

pub(super) fn find_importable_node<'a: 'db, 'db>(
    ctx: &'a AssistContext<'db>,
) -> Option<(ImportAssets<'db>, SyntaxNode, Option<Type<'db>>)> {
    // Deduplicate this with the `expected_type_and_name` logic for completions
    let expected = |expr_or_pat: Either<ast::Expr, ast::Pat>| match expr_or_pat {
        Either::Left(expr) => {
            let parent = expr.syntax().parent()?;
            // FIXME: Expand this
            match_ast! {
                match parent {
                    ast::ArgList(list) => {
                        ActiveParameter::at_arg(
                            &ctx.sema,
                            list,
                            expr.syntax().text_range().start(),
                        ).map(|ap| ap.ty)
                    },
                    ast::LetStmt(stmt) => {
                        ctx.sema.type_of_pat(&stmt.pat()?).map(|t| t.original)
                    },
                    _ => None,
                }
            }
        }
        Either::Right(pat) => {
            let parent = pat.syntax().parent()?;
            // FIXME: Expand this
            match_ast! {
                match parent {
                    ast::LetStmt(stmt) => {
                        ctx.sema.type_of_expr(&stmt.initializer()?).map(|t| t.original)
                    },
                    _ => None,
                }
            }
        }
    };

    if let Some(path_under_caret) = ctx.find_node_at_offset_with_descend::<ast::Path>() {
        let expected =
            path_under_caret.top_path().syntax().parent().and_then(Either::cast).and_then(expected);
        ImportAssets::for_exact_path(&path_under_caret, &ctx.sema)
            .map(|it| (it, path_under_caret.syntax().clone(), expected))
    } else if let Some(method_under_caret) =
        ctx.find_node_at_offset_with_descend::<ast::MethodCallExpr>()
    {
        let expected = expected(Either::Left(method_under_caret.clone().into()));
        ImportAssets::for_method_call(&method_under_caret, &ctx.sema)
            .map(|it| (it, method_under_caret.syntax().clone(), expected))
    } else if ctx.find_node_at_offset_with_descend::<ast::Param>().is_some() {
        None
    } else if let Some(pat) = ctx
        .find_node_at_offset_with_descend::<ast::IdentPat>()
        .filter(ast::IdentPat::is_simple_ident)
    {
        let expected = expected(Either::Right(pat.clone().into()));
        ImportAssets::for_ident_pat(&ctx.sema, &pat).map(|it| (it, pat.syntax().clone(), expected))
    } else {
        None
    }
}

fn group_label(import_candidate: &ImportCandidate<'_>) -> GroupLabel {
    let name = match import_candidate {
        ImportCandidate::Path(candidate) => format!("Import {}", candidate.name.text()),
        ImportCandidate::TraitAssocItem(candidate) => {
            format!("Import a trait for item {}", candidate.assoc_item_name.text())
        }
        ImportCandidate::TraitMethod(candidate) => {
            format!("Import a trait for method {}", candidate.assoc_item_name.text())
        }
    };
    GroupLabel(name)
}

/// Determine how relevant a given import is in the current context. Higher scores are more
/// relevant.
pub(crate) fn relevance_score(
    ctx: &AssistContext<'_>,
    import: &LocatedImport,
    expected: Option<&Type<'_>>,
    current_module: Option<&Module>,
) -> i32 {
    let mut score = 0;

    let db = ctx.db();

    let item_module = match import.item_to_import {
        hir::ItemInNs::Types(item) | hir::ItemInNs::Values(item) => item.module(db),
        hir::ItemInNs::Macros(makro) => Some(makro.module(db)),
    };

    if let Some(expected) = expected {
        let ty = match import.item_to_import {
            hir::ItemInNs::Types(module_def) | hir::ItemInNs::Values(module_def) => {
                match module_def {
                    hir::ModuleDef::Function(function) => Some(function.ret_type(ctx.db())),
                    hir::ModuleDef::Adt(adt) => Some(match adt {
                        hir::Adt::Struct(it) => it.ty(ctx.db()),
                        hir::Adt::Union(it) => it.ty(ctx.db()),
                        hir::Adt::Enum(it) => it.ty(ctx.db()),
                    }),
                    hir::ModuleDef::Variant(variant) => Some(variant.constructor_ty(ctx.db())),
                    hir::ModuleDef::Const(it) => Some(it.ty(ctx.db())),
                    hir::ModuleDef::Static(it) => Some(it.ty(ctx.db())),
                    hir::ModuleDef::TypeAlias(it) => Some(it.ty(ctx.db())),
                    hir::ModuleDef::BuiltinType(it) => Some(it.ty(ctx.db())),
                    _ => None,
                }
            }
            hir::ItemInNs::Macros(_) => None,
        };
        if let Some(ty) = ty {
            if ty == *expected {
                score = 100000;
            } else if ty.could_unify_with(ctx.db(), expected) {
                score = 10000;
            }
        }
    }

    match item_module.zip(current_module) {
        // get the distance between the imported path and the current module
        // (prefer items that are more local)
        Some((item_module, current_module)) => {
            score -= module_distance_heuristic(db, current_module, &item_module) as i32;
        }

        // could not find relevant modules, so just use the length of the path as an estimate
        None => return -(2 * import.import_path.len() as i32),
    }

    score
}

/// A heuristic that gives a higher score to modules that are more separated.
fn module_distance_heuristic(db: &dyn HirDatabase, current: &Module, item: &Module) -> usize {
    // get the path starting from the item to the respective crate roots
    let mut current_path = current.path_to_root(db);
    let mut item_path = item.path_to_root(db);

    // we want paths going from the root to the item
    current_path.reverse();
    item_path.reverse();

    // length of the common prefix of the two paths
    let prefix_length = current_path.iter().zip(&item_path).take_while(|(a, b)| a == b).count();

    // how many modules differ between the two paths (all modules, removing any duplicates)
    let distinct_length = current_path.len() + item_path.len() - 2 * prefix_length;

    // cost of importing from another crate
    let crate_boundary_cost = if current.krate() == item.krate() {
        0
    } else if item.krate().origin(db).is_local() {
        2
    } else if item.krate().is_builtin(db) {
        3
    } else {
        4
    };

    distinct_length + crate_boundary_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    use hir::{FileRange, Semantics};
    use ide_db::{RootDatabase, assists::AssistResolveStrategy};
    use test_fixture::WithFixture;

    use crate::tests::{
        TEST_CONFIG, check_assist, check_assist_by_label, check_assist_not_applicable,
        check_assist_target,
    };

    fn check_auto_import_order(before: &str, order: &[&str]) {
        let (db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(before);
        let frange = FileRange { file_id, range: range_or_offset.into() };

        let sema = Semantics::new(&db);
        let config = TEST_CONFIG;
        let ctx = AssistContext::new(sema, &config, frange);
        let mut acc = Assists::new(&ctx, AssistResolveStrategy::All);
        auto_import(&mut acc, &ctx);
        let assists = acc.finish();

        let labels = assists.iter().map(|assist| assist.label.to_string()).collect::<Vec<_>>();

        assert_eq!(labels, order);
    }

    #[test]
    fn ignore_parameter_name() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod foo {
                pub mod bar {}
            }

            fn foo(bar$0: &str) {}
            ",
        );
    }

    #[test]
    fn prefer_shorter_paths() {
        let before = r"
//- /main.rs crate:main deps:foo,bar
HashMap$0::new();

//- /lib.rs crate:foo
pub mod collections { pub struct HashMap; }

//- /lib.rs crate:bar
pub mod collections { pub mod hash_map { pub struct HashMap; } }
        ";

        check_auto_import_order(
            before,
            &["Import `foo::collections::HashMap`", "Import `bar::collections::hash_map::HashMap`"],
        )
    }

    #[test]
    fn prefer_same_crate() {
        let before = r"
//- /main.rs crate:main deps:foo
HashMap$0::new();

mod collections {
    pub mod hash_map {
        pub struct HashMap;
    }
}

//- /lib.rs crate:foo
pub struct HashMap;
        ";

        check_auto_import_order(
            before,
            &["Import `collections::hash_map::HashMap`", "Import `foo::HashMap`"],
        )
    }

    #[test]
    fn prefer_workspace() {
        let before = r"
//- /main.rs crate:main deps:foo,bar
HashMap$0::new();

//- /lib.rs crate:foo
pub mod module {
    pub struct HashMap;
}

//- /lib.rs crate:bar library
pub struct HashMap;
        ";

        check_auto_import_order(before, &["Import `foo::module::HashMap`", "Import `bar::HashMap`"])
    }

    #[test]
    fn prefer_non_local_over_long_path() {
        let before = r"
//- /main.rs crate:main deps:foo,bar
HashMap$0::new();

//- /lib.rs crate:foo
pub mod deeply {
    pub mod nested {
        pub mod module {
            pub struct HashMap;
        }
    }
}

//- /lib.rs crate:bar library
pub struct HashMap;
        ";

        check_auto_import_order(
            before,
            &["Import `bar::HashMap`", "Import `foo::deeply::nested::module::HashMap`"],
        )
    }

    #[test]
    fn not_applicable_if_scope_inside_macro() {
        check_assist_not_applicable(
            auto_import,
            r"
mod bar {
    pub struct Baz;
}
macro_rules! foo {
    ($it:ident) => {
        mod __ {
            fn __(x: $it) {}
        }
    };
}
foo! {
    Baz$0
}
",
        );
    }

    #[test]
    fn applicable_in_attributes() {
        check_assist(
            auto_import,
            r"
//- proc_macros: identity
#[proc_macros::identity]
mod foo {
    mod bar {
        const _: Baz$0 = ();
    }
}
mod baz {
    pub struct Baz;
}
",
            r"
#[proc_macros::identity]
mod foo {
    mod bar {
        use crate::baz::Baz;

        const _: Baz = ();
    }
}
mod baz {
    pub struct Baz;
}
",
        );
    }

    #[test]
    fn applicable_when_found_an_import_partial() {
        check_assist(
            auto_import,
            r"
            mod std {
                pub mod fmt {
                    pub struct Formatter;
                }
            }

            use std::fmt;

            $0Formatter
            ",
            r"
            mod std {
                pub mod fmt {
                    pub struct Formatter;
                }
            }

            use std::fmt::{self, Formatter};

            Formatter
            ",
        );
    }

    #[test]
    fn applicable_when_found_an_import() {
        check_assist(
            auto_import,
            r"
            $0PubStruct

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
            r"
            use PubMod::PubStruct;

            PubStruct

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn applicable_when_found_an_import_in_macros() {
        check_assist(
            auto_import,
            r"
            macro_rules! foo {
                ($i:ident) => { fn foo(a: $i) {} }
            }
            foo!(Pub$0Struct);

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
            r"
            use PubMod::PubStruct;

            macro_rules! foo {
                ($i:ident) => { fn foo(a: $i) {} }
            }
            foo!(PubStruct);

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn applicable_when_found_multiple_imports() {
        check_assist(
            auto_import,
            r"
            PubSt$0ruct

            pub mod PubMod1 {
                pub struct PubStruct;
            }
            pub mod PubMod2 {
                pub struct PubStruct;
            }
            pub mod PubMod3 {
                pub struct PubStruct;
            }
            ",
            r"
            use PubMod1::PubStruct;

            PubStruct

            pub mod PubMod1 {
                pub struct PubStruct;
            }
            pub mod PubMod2 {
                pub struct PubStruct;
            }
            pub mod PubMod3 {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_already_imported_types() {
        check_assist_not_applicable(
            auto_import,
            r"
            use PubMod::PubStruct;

            PubStruct$0

            pub mod PubMod {
                pub struct PubStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_types_with_private_paths() {
        check_assist_not_applicable(
            auto_import,
            r"
            PrivateStruct$0

            pub mod PubMod {
                struct PrivateStruct;
            }
            ",
        );
    }

    #[test]
    fn not_applicable_when_no_imports_found() {
        check_assist_not_applicable(
            auto_import,
            "
            PubStruct$0",
        );
    }

    #[test]
    fn function_import() {
        check_assist(
            auto_import,
            r"
            test_function$0

            pub mod PubMod {
                pub fn test_function() {};
            }
            ",
            r"
            use PubMod::test_function;

            test_function

            pub mod PubMod {
                pub fn test_function() {};
            }
            ",
        );
    }

    #[test]
    fn macro_import() {
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:crate_with_macro
#[macro_export]
macro_rules! foo {
    () => ()
}

//- /main.rs crate:main deps:crate_with_macro
fn main() {
    foo$0
}
",
            r"use crate_with_macro::foo;

fn main() {
    foo
}
",
        );
    }

    #[test]
    fn auto_import_target() {
        check_assist_target(
            auto_import,
            r"
            struct AssistInfo {
                group_label: Option<$0GroupLabel>,
            }

            mod m { pub struct GroupLabel; }
            ",
            "GroupLabel",
        )
    }

    #[test]
    fn not_applicable_when_path_start_is_imported() {
        check_assist_not_applicable(
            auto_import,
            r"
            pub mod mod1 {
                pub mod mod2 {
                    pub mod mod3 {
                        pub struct TestStruct;
                    }
                }
            }

            use mod1::mod2;
            fn main() {
                mod2::mod3::TestStruct$0
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_function() {
        check_assist_not_applicable(
            auto_import,
            r"
            pub mod test_mod {
                pub fn test_function() {}
            }

            use test_mod::test_function;
            fn main() {
                test_function$0
            }
            ",
        );
    }

    #[test]
    fn associated_struct_function() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    pub fn test_function() {}
                }
            }

            fn main() {
                TestStruct::test_function$0
            }
            ",
            r"
            use test_mod::TestStruct;

            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    pub fn test_function() {}
                }
            }

            fn main() {
                TestStruct::test_function
            }
            ",
        );
    }

    #[test]
    fn associated_struct_const() {
        check_assist(
            auto_import,
            r"
            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                TestStruct::TEST_CONST$0
            }
            ",
            r"
            use test_mod::TestStruct;

            mod test_mod {
                pub struct TestStruct {}
                impl TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                TestStruct::TEST_CONST
            }
            ",
        );
    }

    #[test]
    fn associated_trait_function() {
        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function$0
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function
            }
            ",
            "Import `test_mod::TestTrait`",
        );

        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function$0
            }
            ",
            r"
            use test_mod::TestTrait as _;

            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }

            fn main() {
                test_mod::TestStruct::test_function
            }
            ",
            "Import `test_mod::TestTrait as _`",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait_for_function() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub trait TestTrait2 {
                    fn test_function();
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    fn test_function() {}
                }
                impl TestTrait for TestEnum {
                    fn test_function() {}
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                test_mod::TestEnum::test_function$0;
            }
            ",
        )
    }

    #[test]
    fn associated_trait_const() {
        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST$0
            }
            ",
            r"
            use test_mod::TestTrait as _;

            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST
            }
            ",
            "Import `test_mod::TestTrait as _`",
        );

        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST$0
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const TEST_CONST: u8 = 42;
                }
            }

            fn main() {
                test_mod::TestStruct::TEST_CONST
            }
            ",
            "Import `test_mod::TestTrait`",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait_for_const() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    const TEST_CONST: u8;
                }
                pub trait TestTrait2 {
                    const TEST_CONST: f64;
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    const TEST_CONST: f64 = 42.0;
                }
                impl TestTrait for TestEnum {
                    const TEST_CONST: u8 = 42;
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                test_mod::TestEnum::TEST_CONST$0;
            }
            ",
        )
    }

    #[test]
    fn trait_method() {
        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_meth$0od()
            }
            ",
            r"
            use test_mod::TestTrait as _;

            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_method()
            }
            ",
            "Import `test_mod::TestTrait as _`",
        );

        check_assist_by_label(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_meth$0od()
            }
            ",
            r"
            use test_mod::TestTrait;

            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }

            fn main() {
                let test_struct = test_mod::TestStruct {};
                test_struct.test_method()
            }
            ",
            "Import `test_mod::TestTrait`",
        );
    }

    #[test]
    fn trait_method_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_meth$0od()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_method()
            }
            ",
            "Import `dep::test_mod::TestTrait as _`",
        );

        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_meth$0od()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_method()
            }
            ",
            "Import `dep::test_mod::TestTrait`",
        );
    }

    #[test]
    fn assoc_fn_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                dep::test_mod::TestStruct::test_function
            }
            ",
            "Import `dep::test_mod::TestTrait as _`",
        );

        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                dep::test_mod::TestStruct::test_function
            }
            ",
            "Import `dep::test_mod::TestTrait`",
        );
    }

    #[test]
    fn assoc_const_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::CONST$0
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    const CONST: bool;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const CONST: bool = true;
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                dep::test_mod::TestStruct::CONST
            }
            ",
            "Import `dep::test_mod::TestTrait as _`",
        );

        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::CONST$0
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    const CONST: bool;
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    const CONST: bool = true;
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                dep::test_mod::TestStruct::CONST
            }
            ",
            "Import `dep::test_mod::TestTrait`",
        );
    }

    #[test]
    fn assoc_fn_as_method_cross_crate() {
        check_assist_not_applicable(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_func$0tion()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
        );
    }

    #[test]
    fn private_trait_cross_crate() {
        check_assist_not_applicable(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_meth$0od()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
        );
    }

    #[test]
    fn not_applicable_for_imported_trait_for_method() {
        check_assist_not_applicable(
            auto_import,
            r"
            mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub trait TestTrait2 {
                    fn test_method(&self);
                }
                pub enum TestEnum {
                    One,
                    Two,
                }
                impl TestTrait2 for TestEnum {
                    fn test_method(&self) {}
                }
                impl TestTrait for TestEnum {
                    fn test_method(&self) {}
                }
            }

            use test_mod::TestTrait2;
            fn main() {
                let one = test_mod::TestEnum::One;
                one.test$0_method();
            }
            ",
        )
    }

    #[test]
    fn dep_import() {
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub struct Struct;

//- /main.rs crate:main deps:dep
fn main() {
    Struct$0
}
",
            r"use dep::Struct;

fn main() {
    Struct
}
",
        );
    }

    #[test]
    fn whole_segment() {
        // Tests that only imports whose last segment matches the identifier get suggested.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub mod fmt {
    pub trait Display {}
}

pub fn panic_fmt() {}

//- /main.rs crate:main deps:dep
struct S;

impl f$0mt::Display for S {}
",
            r"use dep::fmt;

struct S;

impl fmt::Display for S {}
",
        );
    }

    #[test]
    fn macro_generated() {
        // Tests that macro-generated items are suggested from external crates.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
macro_rules! mac {
    () => {
        pub struct Cheese;
    };
}

mac!();

//- /main.rs crate:main deps:dep
fn main() {
    Cheese$0;
}
",
            r"use dep::Cheese;

fn main() {
    Cheese;
}
",
        );
    }

    #[test]
    fn casing() {
        // Tests that differently cased names don't interfere and we only suggest the matching one.
        check_assist(
            auto_import,
            r"
//- /lib.rs crate:dep
pub struct FMT;
pub struct fmt;

//- /main.rs crate:main deps:dep
fn main() {
    FMT$0;
}
",
            r"use dep::FMT;

fn main() {
    FMT;
}
",
        );
    }

    #[test]
    fn inner_items() {
        check_assist(
            auto_import,
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    fn bar() {
        Foo$0;
        println!("Hallo");
    }
}
"#,
            r#"
mod baz {
    pub struct Foo {}
}

mod bar {
    use crate::baz::Foo;

    fn bar() {
        Foo;
        println!("Hallo");
    }
}
"#,
        );
    }

    #[test]
    fn uses_abs_path_with_extern_crate_clash() {
        cov_mark::check!(ambiguous_crate_start);
        check_assist(
            auto_import,
            r#"
//- /main.rs crate:main deps:foo
mod foo {}

const _: () = {
    Foo$0
};
//- /foo.rs crate:foo
pub struct Foo
"#,
            r#"
use ::foo::Foo;

mod foo {}

const _: () = {
    Foo
};
"#,
        );
    }

    #[test]
    fn works_on_ident_patterns() {
        check_assist(
            auto_import,
            r#"
mod foo {
    pub struct Foo {}
}
fn foo() {
    let Foo$0;
}
"#,
            r#"
use foo::Foo;

mod foo {
    pub struct Foo {}
}
fn foo() {
    let Foo;
}
"#,
        );
    }

    #[test]
    fn works_in_derives() {
        check_assist(
            auto_import,
            r#"
//- minicore:derive
mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(Copy$0)]
struct Foo;
"#,
            r#"
use foo::Copy;

mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(Copy)]
struct Foo;
"#,
        );
    }

    #[test]
    fn works_in_use_start() {
        check_assist(
            auto_import,
            r#"
mod bar {
    pub mod foo {
        pub struct Foo;
    }
}
use foo$0::Foo;
"#,
            r#"
mod bar {
    pub mod foo {
        pub struct Foo;
    }
}
use bar::foo;
use foo::Foo;
"#,
        );
    }

    #[test]
    fn not_applicable_in_non_start_use() {
        check_assist_not_applicable(
            auto_import,
            r"
mod bar {
    pub mod foo {
        pub struct Foo;
    }
}
use foo::Foo$0;
",
        );
    }

    #[test]
    fn considers_pub_crate() {
        check_assist(
            auto_import,
            r#"
mod foo {
    pub struct Foo;
}

pub(crate) use self::foo::*;

mod bar {
    fn main() {
        Foo$0;
    }
}
"#,
            r#"
mod foo {
    pub struct Foo;
}

pub(crate) use self::foo::*;

mod bar {
    use crate::Foo;

    fn main() {
        Foo;
    }
}
"#,
        );
    }

    #[test]
    fn local_inline_import_has_alias() {
        // FIXME wrong import
        check_assist(
            auto_import,
            r#"
struct S<T>(T);
use S as IoResult;

mod foo {
    pub fn bar() -> S$0<()> {}
}
"#,
            r#"
struct S<T>(T);
use S as IoResult;

mod foo {
    use crate::S;

    pub fn bar() -> S<()> {}
}
"#,
        );
    }

    #[test]
    fn alias_local() {
        // FIXME wrong import
        check_assist(
            auto_import,
            r#"
struct S<T>(T);
use S as IoResult;

mod foo {
    pub fn bar() -> IoResult$0<()> {}
}
"#,
            r#"
struct S<T>(T);
use S as IoResult;

mod foo {
    use crate::S;

    pub fn bar() -> IoResult<()> {}
}
"#,
        );
    }

    #[test]
    fn preserve_raw_identifiers_strict() {
        check_assist(
            auto_import,
            r"
            r#as$0

            pub mod ffi_mod {
                pub fn r#as() {};
            }
            ",
            r"
            use ffi_mod::r#as;

            r#as

            pub mod ffi_mod {
                pub fn r#as() {};
            }
            ",
        );
    }

    #[test]
    fn preserve_raw_identifiers_reserved() {
        check_assist(
            auto_import,
            r"
            r#abstract$0

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
            r"
            use ffi_mod::r#abstract;

            r#abstract

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
        );
    }

    #[test]
    fn prefers_type_match() {
        check_assist(
            auto_import,
            r"
mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: sync::atomic::Ordering) {}
fn main() {
    takes_ordering(Ordering$0);
}
",
            r"
use sync::atomic::Ordering;

mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: sync::atomic::Ordering) {}
fn main() {
    takes_ordering(Ordering);
}
",
        );
        check_assist(
            auto_import,
            r"
mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: cmp::Ordering) {}
fn main() {
    takes_ordering(Ordering$0);
}
",
            r"
use cmp::Ordering;

mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: cmp::Ordering) {}
fn main() {
    takes_ordering(Ordering);
}
",
        );
    }

    #[test]
    fn prefers_type_match2() {
        check_assist(
            auto_import,
            r"
mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: sync::atomic::Ordering) {}
fn main() {
    takes_ordering(Ordering$0::V);
}
",
            r"
use sync::atomic::Ordering;

mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: sync::atomic::Ordering) {}
fn main() {
    takes_ordering(Ordering::V);
}
",
        );
        check_assist(
            auto_import,
            r"
mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: cmp::Ordering) {}
fn main() {
    takes_ordering(Ordering$0::V);
}
",
            r"
use cmp::Ordering;

mod sync { pub mod atomic { pub enum Ordering { V } } }
mod cmp { pub enum Ordering { V } }
fn takes_ordering(_: cmp::Ordering) {}
fn main() {
    takes_ordering(Ordering::V);
}
",
        );
    }

    #[test]
    fn carries_cfg_attr() {
        check_assist(
            auto_import,
            r#"
mod m {
    pub struct S;
}

#[cfg(test)]
fn foo(_: S$0) {}
"#,
            r#"
#[cfg(test)]
use m::S;

mod m {
    pub struct S;
}

#[cfg(test)]
fn foo(_: S) {}
"#,
        );
    }
}
