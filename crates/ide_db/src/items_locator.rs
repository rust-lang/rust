//! This module contains an import search functionality that is provided to the assists module.
//! Later, this should be moved away to a separate crate that is accessible from the assists module.

use either::Either;
use hir::{
    import_map::{self, ImportKind},
    AsAssocItem, Crate, ItemInNs, ModuleDef, Semantics,
};
use syntax::{ast, AstNode, SyntaxKind::NAME};

use crate::{
    defs::{Definition, NameClass},
    helpers::import_assets::NameToImport,
    symbol_index::{self, FileSymbol},
    RootDatabase,
};
use rustc_hash::FxHashSet;

pub(crate) const DEFAULT_QUERY_SEARCH_LIMIT: usize = 40;

/// TODO kb docs here and around + update the module doc
#[derive(Debug, Clone, Copy)]
pub enum AssocItemSearch {
    Include,
    Exclude,
    AssocItemsOnly,
}

pub fn locate_for_name(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    name: NameToImport,
    assoc_item_search: AssocItemSearch,
    limit: Option<usize>,
) -> FxHashSet<ItemInNs> {
    let _p = profile::span("locate_for_name").detail(|| {
        format!(
            "Name: {} ({:?}), crate: {:?}, limit: {:?}",
            name.text(),
            assoc_item_search,
            krate.display_name(sema.db).map(|name| name.to_string()),
            limit,
        )
    });

    let (mut local_query, mut external_query) = match name {
        NameToImport::Exact(exact_name) => {
            let mut local_query = symbol_index::Query::new(exact_name.clone());
            local_query.exact();

            let external_query = import_map::Query::new(exact_name)
                .name_only()
                .search_mode(import_map::SearchMode::Equals)
                .case_sensitive();

            (local_query, external_query)
        }
        NameToImport::Fuzzy(fuzzy_search_string) => {
            let mut external_query = import_map::Query::new(fuzzy_search_string.clone())
                .search_mode(import_map::SearchMode::Fuzzy)
                .name_only();
            match assoc_item_search {
                AssocItemSearch::Include => {}
                AssocItemSearch::Exclude => {
                    external_query = external_query.exclude_import_kind(ImportKind::AssociatedItem);
                }
                AssocItemSearch::AssocItemsOnly => {
                    external_query = external_query.assoc_items_only();
                }
            }

            (symbol_index::Query::new(fuzzy_search_string), external_query)
        }
    };

    if let Some(limit) = limit {
        external_query = external_query.limit(limit);
        local_query.limit(limit);
    }

    find_items(sema, krate, assoc_item_search, local_query, external_query)
}

fn find_items(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    assoc_item_search: AssocItemSearch,
    local_query: symbol_index::Query,
    external_query: import_map::Query,
) -> FxHashSet<ItemInNs> {
    let _p = profile::span("find_items");
    let db = sema.db;

    let external_importables =
        krate.query_external_importables(db, external_query).map(|external_importable| {
            match external_importable {
                Either::Left(module_def) => ItemInNs::from(module_def),
                Either::Right(macro_def) => ItemInNs::from(macro_def),
            }
        });

    // Query the local crate using the symbol index.
    let local_results = symbol_index::crate_symbols(db, krate.into(), local_query)
        .into_iter()
        .filter_map(|local_candidate| get_name_definition(sema, &local_candidate))
        .filter_map(|name_definition_to_import| match name_definition_to_import {
            Definition::ModuleDef(module_def) => Some(ItemInNs::from(module_def)),
            Definition::Macro(macro_def) => Some(ItemInNs::from(macro_def)),
            _ => None,
        });

    external_importables
        .chain(local_results)
        .filter(move |&item| match assoc_item_search {
            AssocItemSearch::Include => true,
            AssocItemSearch::Exclude => !is_assoc_item(item, sema.db),
            AssocItemSearch::AssocItemsOnly => is_assoc_item(item, sema.db),
        })
        .collect()
}

fn get_name_definition(
    sema: &Semantics<'_, RootDatabase>,
    import_candidate: &FileSymbol,
) -> Option<Definition> {
    let _p = profile::span("get_name_definition");
    let file_id = import_candidate.file_id;

    let candidate_node = import_candidate.ptr.to_node(sema.parse(file_id).syntax());
    let candidate_name_node = if candidate_node.kind() != NAME {
        candidate_node.children().find(|it| it.kind() == NAME)?
    } else {
        candidate_node
    };
    let name = ast::Name::cast(candidate_name_node)?;
    NameClass::classify(sema, &name)?.defined(sema.db)
}

fn is_assoc_item(item: ItemInNs, db: &RootDatabase) -> bool {
    item.as_module_def_id()
        .and_then(|module_def_id| ModuleDef::from(module_def_id).as_assoc_item(db))
        .is_some()
}
