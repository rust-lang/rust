//! See [`import_on_the_fly`].
use hir::{ItemInNs, ModuleDef};
use ide_db::imports::{
    import_assets::{ImportAssets, LocatedImport},
    insert_use::ImportScope,
};
use itertools::Itertools;
use syntax::{ast, AstNode, SyntaxNode, T};

use crate::{
    context::{
        CompletionContext, DotAccess, PathCompletionCtx, PathKind, PatternContext, Qualified,
        TypeLocation,
    },
    render::{render_resolution_with_import, render_resolution_with_import_pat, RenderContext},
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
// It is possible to configure how use-trees are merged with the `imports.granularity.group` setting.
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
pub(crate) fn import_on_the_fly_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx,
) -> Option<()> {
    if !ctx.config.enable_imports_on_the_fly {
        return None;
    }
    let qualified = match path_ctx {
        PathCompletionCtx {
            kind:
                PathKind::Expr { .. }
                | PathKind::Type { .. }
                | PathKind::Attr { .. }
                | PathKind::Derive { .. }
                | PathKind::Item { .. }
                | PathKind::Pat { .. },
            qualified,
            ..
        } => qualified,
        _ => return None,
    };
    let potential_import_name = import_name(ctx);
    let qualifier = match qualified {
        Qualified::With { path, .. } => Some(path.clone()),
        _ => None,
    };
    let import_assets = import_assets_for_path(ctx, &potential_import_name, qualifier.clone())?;

    import_on_the_fly(
        acc,
        ctx,
        path_ctx,
        import_assets,
        qualifier.map(|it| it.syntax().clone()).or_else(|| ctx.original_token.parent())?,
        potential_import_name,
    )
}

pub(crate) fn import_on_the_fly_pat(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
) -> Option<()> {
    if !ctx.config.enable_imports_on_the_fly {
        return None;
    }
    if let PatternContext { record_pat: Some(_), .. } = pattern_ctx {
        return None;
    }

    let potential_import_name = import_name(ctx);
    let import_assets = import_assets_for_path(ctx, &potential_import_name, None)?;

    import_on_the_fly_pat_(
        acc,
        ctx,
        pattern_ctx,
        import_assets,
        ctx.original_token.parent()?,
        potential_import_name,
    )
}

pub(crate) fn import_on_the_fly_dot(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess,
) -> Option<()> {
    if !ctx.config.enable_imports_on_the_fly {
        return None;
    }
    let receiver = dot_access.receiver.as_ref()?;
    let ty = dot_access.receiver_ty.as_ref()?;
    let potential_import_name = import_name(ctx);
    let import_assets = ImportAssets::for_fuzzy_method_call(
        ctx.module,
        ty.original.clone(),
        potential_import_name.clone(),
        receiver.syntax().clone(),
    )?;

    import_on_the_fly_method(
        acc,
        ctx,
        dot_access,
        import_assets,
        receiver.syntax().clone(),
        potential_import_name,
    )
}

fn import_on_the_fly(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { kind, .. }: &PathCompletionCtx,
    import_assets: ImportAssets,
    position: SyntaxNode,
    potential_import_name: String,
) -> Option<()> {
    let _p = profile::span("import_on_the_fly").detail(|| potential_import_name.clone());

    if ImportScope::find_insert_use_container(&position, &ctx.sema).is_none() {
        return None;
    }

    let ns_filter = |import: &LocatedImport| {
        match (kind, import.original_item) {
            // Aren't handled in flyimport
            (PathKind::Vis { .. } | PathKind::Use, _) => false,
            // modules are always fair game
            (_, ItemInNs::Types(hir::ModuleDef::Module(_))) => true,
            // and so are macros(except for attributes)
            (
                PathKind::Expr { .. }
                | PathKind::Type { .. }
                | PathKind::Item { .. }
                | PathKind::Pat { .. },
                ItemInNs::Macros(mac),
            ) => mac.is_fn_like(ctx.db),
            (PathKind::Item { .. }, ..) => false,

            (PathKind::Expr { .. }, ItemInNs::Types(_) | ItemInNs::Values(_)) => true,

            (PathKind::Pat { .. }, ItemInNs::Types(_)) => true,
            (PathKind::Pat { .. }, ItemInNs::Values(def)) => {
                matches!(def, hir::ModuleDef::Const(_))
            }

            (PathKind::Type { location }, ItemInNs::Types(ty)) => {
                if matches!(location, TypeLocation::TypeBound) {
                    matches!(ty, ModuleDef::Trait(_))
                } else {
                    true
                }
            }
            (PathKind::Type { .. }, ItemInNs::Values(_)) => false,

            (PathKind::Attr { .. }, ItemInNs::Macros(mac)) => mac.is_attr(ctx.db),
            (PathKind::Attr { .. }, _) => false,

            (PathKind::Derive { existing_derives }, ItemInNs::Macros(mac)) => {
                mac.is_derive(ctx.db) && !existing_derives.contains(&mac)
            }
            (PathKind::Derive { .. }, _) => false,
        }
    };
    let user_input_lowercased = potential_import_name.to_lowercase();

    acc.add_all(
        import_assets
            .search_for_imports(
                &ctx.sema,
                ctx.config.insert_use.prefix_kind,
                ctx.config.prefer_no_std,
            )
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
                render_resolution_with_import(RenderContext::new(ctx), path_ctx, import)
            })
            .map(|builder| builder.build()),
    );
    Some(())
}

fn import_on_the_fly_pat_(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
    import_assets: ImportAssets,
    position: SyntaxNode,
    potential_import_name: String,
) -> Option<()> {
    let _p = profile::span("import_on_the_fly_pat").detail(|| potential_import_name.clone());

    if ImportScope::find_insert_use_container(&position, &ctx.sema).is_none() {
        return None;
    }

    let ns_filter = |import: &LocatedImport| match import.original_item {
        ItemInNs::Macros(mac) => mac.is_fn_like(ctx.db),
        ItemInNs::Types(_) => true,
        ItemInNs::Values(def) => matches!(def, hir::ModuleDef::Const(_)),
    };
    let user_input_lowercased = potential_import_name.to_lowercase();

    acc.add_all(
        import_assets
            .search_for_imports(
                &ctx.sema,
                ctx.config.insert_use.prefix_kind,
                ctx.config.prefer_no_std,
            )
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
                render_resolution_with_import_pat(RenderContext::new(ctx), pattern_ctx, import)
            })
            .map(|builder| builder.build()),
    );
    Some(())
}

fn import_on_the_fly_method(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess,
    import_assets: ImportAssets,
    position: SyntaxNode,
    potential_import_name: String,
) -> Option<()> {
    let _p = profile::span("import_on_the_fly_method").detail(|| potential_import_name.clone());

    if ImportScope::find_insert_use_container(&position, &ctx.sema).is_none() {
        return None;
    }

    let user_input_lowercased = potential_import_name.to_lowercase();

    import_assets
        .search_for_imports(&ctx.sema, ctx.config.insert_use.prefix_kind, ctx.config.prefer_no_std)
        .into_iter()
        .filter(|import| {
            !ctx.is_item_hidden(&import.item_to_import)
                && !ctx.is_item_hidden(&import.original_item)
        })
        .sorted_by_key(|located_import| {
            compute_fuzzy_completion_order_key(&located_import.import_path, &user_input_lowercased)
        })
        .for_each(|import| match import.original_item {
            ItemInNs::Values(hir::ModuleDef::Function(f)) => {
                acc.add_method_with_import(ctx, dot_access, f, import);
            }
            _ => (),
        });
    Some(())
}

fn import_name(ctx: &CompletionContext<'_>) -> String {
    let token_kind = ctx.token.kind();
    if matches!(token_kind, T![.] | T![::]) {
        String::new()
    } else {
        ctx.token.to_string()
    }
}

fn import_assets_for_path(
    ctx: &CompletionContext<'_>,
    potential_import_name: &str,
    qualifier: Option<ast::Path>,
) -> Option<ImportAssets> {
    let fuzzy_name_length = potential_import_name.len();
    let mut assets_for_path = ImportAssets::for_fuzzy_path(
        ctx.module,
        qualifier,
        potential_import_name.to_owned(),
        &ctx.sema,
        ctx.token.parent()?,
    )?;
    if fuzzy_name_length < 3 {
        cov_mark::hit!(flyimport_exact_on_short_path);
        assets_for_path.path_fuzzy_name_to_exact(false);
    }
    Some(assets_for_path)
}

fn compute_fuzzy_completion_order_key(
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
