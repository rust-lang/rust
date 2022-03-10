//! See [`import_on_the_fly`].
use hir::ItemInNs;
use ide_db::imports::{
    import_assets::{ImportAssets, ImportCandidate, LocatedImport},
    insert_use::ImportScope,
};
use itertools::Itertools;
use syntax::{AstNode, SyntaxNode, T};

use crate::{
    context::{CompletionContext, PathKind},
    render::{render_resolution_with_import, RenderContext},
    ImportEdit,
};

use super::Completions;

// Feature: Completion With Autoimport
//
// When completing names in the current scope, proposes additional imports from other modules or crates,
// if they can be qualified in the scope, and their name contains all symbols from the completion input.
//
// To be considered applicable, the name must contain all input symbols in the given order, not necessarily adjacent.
// If any input symbol is not lowercased, the name must contain all symbols in exact case; otherwise the containing is checked case-insensitively.
//
// ```
// fn main() {
//     pda$0
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
// Also completes associated items, that require trait imports.
// If any unresolved and/or partially-qualified path precedes the input, it will be taken into account.
// Currently, only the imports with their import path ending with the whole qualifier will be proposed
// (no fuzzy matching for qualifier).
//
// ```
// mod foo {
//     pub mod bar {
//         pub struct Item;
//
//         impl Item {
//             pub const TEST_ASSOC: usize = 3;
//         }
//     }
// }
//
// fn main() {
//     bar::Item::TEST_A$0
// }
// ```
// ->
// ```
// use foo::bar;
//
// mod foo {
//     pub mod bar {
//         pub struct Item;
//
//         impl Item {
//             pub const TEST_ASSOC: usize = 3;
//         }
//     }
// }
//
// fn main() {
//     bar::Item::TEST_ASSOC
// }
// ```
//
// NOTE: currently, if an assoc item comes from a trait that's not currently imported, and it also has an unresolved and/or partially-qualified path,
// no imports will be proposed.
//
// .Fuzzy search details
//
// To avoid an excessive amount of the results returned, completion input is checked for inclusion in the names only
// (i.e. in `HashMap` in the `std::collections::HashMap` path).
// For the same reasons, avoids searching for any path imports for inputs with their length less than 2 symbols
// (but shows all associated items for any input length).
//
// .Import configuration
//
// It is possible to configure how use-trees are merged with the `importMergeBehavior` setting.
// Mimics the corresponding behavior of the `Auto Import` feature.
//
// .LSP and performance implications
//
// The feature is enabled only if the LSP client supports LSP protocol version 3.16+ and reports the `additionalTextEdits`
// (case-sensitive) resolve client capability in its client capabilities.
// This way the server is able to defer the costly computations, doing them for a selected completion item only.
// For clients with no such support, all edits have to be calculated on the completion request, including the fuzzy search completion ones,
// which might be slow ergo the feature is automatically disabled.
//
// .Feature toggle
//
// The feature can be forcefully turned off in the settings with the `rust-analyzer.completion.autoimport.enable` flag.
// Note that having this flag set to `true` does not guarantee that the feature is enabled: your client needs to have the corresponding
// capability enabled.
pub(crate) fn import_on_the_fly(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !ctx.config.enable_imports_on_the_fly {
        return None;
    }
    if matches!(ctx.path_kind(), Some(PathKind::Vis { .. } | PathKind::Use))
        || ctx.is_path_disallowed()
        || ctx.expects_item()
        || ctx.expects_assoc_item()
        || ctx.expects_variant()
    {
        return None;
    }
    // FIXME: This should be encoded in a different way
    if ctx.pattern_ctx.is_none() && ctx.path_context.is_none() && !ctx.has_dot_receiver() {
        // completion inside `ast::Name` of a item declaration
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

    let _p = profile::span("import_on_the_fly").detail(|| potential_import_name.clone());

    let user_input_lowercased = potential_import_name.to_lowercase();
    let import_assets = import_assets(ctx, potential_import_name)?;
    let import_scope = ImportScope::find_insert_use_container(
        &position_for_import(ctx, Some(import_assets.import_candidate()))?,
        &ctx.sema,
    )?;

    let ns_filter = |import: &LocatedImport| {
        let path_kind = match ctx.path_kind() {
            Some(kind) => kind,
            None => {
                return match import.original_item {
                    ItemInNs::Macros(mac) => mac.is_fn_like(ctx.db),
                    _ => true,
                }
            }
        };
        match (path_kind, import.original_item) {
            // Aren't handled in flyimport
            (PathKind::Vis { .. } | PathKind::Use, _) => false,
            // modules are always fair game
            (_, ItemInNs::Types(hir::ModuleDef::Module(_))) => true,
            // and so are macros(except for attributes)
            (
                PathKind::Expr | PathKind::Type | PathKind::Mac | PathKind::Pat,
                ItemInNs::Macros(mac),
            ) => mac.is_fn_like(ctx.db),
            (PathKind::Mac, _) => true,

            (PathKind::Expr, ItemInNs::Types(_) | ItemInNs::Values(_)) => true,

            (PathKind::Pat, ItemInNs::Types(_)) => true,
            (PathKind::Pat, ItemInNs::Values(def)) => matches!(def, hir::ModuleDef::Const(_)),

            (PathKind::Type, ItemInNs::Types(_)) => true,
            (PathKind::Type, ItemInNs::Values(_)) => false,

            (PathKind::Attr { .. }, ItemInNs::Macros(mac)) => mac.is_attr(ctx.db),
            (PathKind::Attr { .. }, _) => false,

            (PathKind::Derive, ItemInNs::Macros(mac)) => {
                mac.is_derive(ctx.db) && !ctx.existing_derives.contains(&mac)
            }
            (PathKind::Derive, _) => false,
        }
    };

    acc.add_all(
        import_assets
            .search_for_imports(&ctx.sema, ctx.config.insert_use.prefix_kind)
            .into_iter()
            .filter(ns_filter)
            .filter(|import| {
                !ctx.is_item_hidden(&import.item_to_import)
                    && !ctx.is_item_hidden(&import.original_item)
            })
            .sorted_by_key(|located_import| {
                compute_fuzzy_completion_order_key(
                    &located_import.import_path,
                    &user_input_lowercased,
                )
            })
            .filter_map(|import| {
                render_resolution_with_import(
                    RenderContext::new(ctx, false),
                    ImportEdit { import, scope: import_scope.clone() },
                )
            }),
    );
    Some(())
}

pub(crate) fn position_for_import(
    ctx: &CompletionContext,
    import_candidate: Option<&ImportCandidate>,
) -> Option<SyntaxNode> {
    Some(
        match import_candidate {
            Some(ImportCandidate::Path(_)) => ctx.name_syntax.as_ref()?.syntax(),
            Some(ImportCandidate::TraitAssocItem(_)) => ctx.path_qual()?.syntax(),
            Some(ImportCandidate::TraitMethod(_)) => ctx.dot_receiver()?.syntax(),
            None => return ctx.original_token.parent(),
        }
        .clone(),
    )
}

fn import_assets(ctx: &CompletionContext, fuzzy_name: String) -> Option<ImportAssets> {
    let current_module = ctx.module?;
    if let Some(dot_receiver) = ctx.dot_receiver() {
        ImportAssets::for_fuzzy_method_call(
            current_module,
            ctx.sema.type_of_expr(dot_receiver)?.original,
            fuzzy_name,
            dot_receiver.syntax().clone(),
        )
    } else {
        let fuzzy_name_length = fuzzy_name.len();
        let mut assets_for_path = ImportAssets::for_fuzzy_path(
            current_module,
            ctx.path_qual().cloned(),
            fuzzy_name,
            &ctx.sema,
            ctx.token.parent()?,
        )?;
        if fuzzy_name_length < 3 {
            cov_mark::hit!(flyimport_exact_on_short_path);
            assets_for_path.path_fuzzy_name_to_exact(false);
        }
        Some(assets_for_path)
    }
}

pub(crate) fn compute_fuzzy_completion_order_key(
    proposed_mod_path: &hir::ModPath,
    user_input_lowercased: &str,
) -> usize {
    cov_mark::hit!(certain_fuzzy_order_test);
    let import_name = match proposed_mod_path.segments().last() {
        Some(name) => name.to_smol_str().to_lowercase(),
        None => return usize::MAX,
    };
    match import_name.match_indices(user_input_lowercased).next() {
        Some((first_matching_index, _)) => first_matching_index,
        None => usize::MAX,
    }
}
