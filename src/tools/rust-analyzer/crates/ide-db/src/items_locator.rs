//! This module has the functionality to search the project and its dependencies for a certain item,
//! by its name and a few criteria.
//! The main reason for this module to exist is the fact that project's items and dependencies' items
//! are located in different caches, with different APIs.
use std::ops::ControlFlow;

use either::Either;
use hir::{Complete, Crate, ItemInNs, Module, import_map};

use crate::{
    RootDatabase,
    imports::import_assets::NameToImport,
    symbol_index::{self, SymbolsDatabase as _},
};

/// A value to use, when uncertain which limit to pick.
pub const DEFAULT_QUERY_SEARCH_LIMIT: usize = 100;

pub use import_map::AssocSearchMode;

// FIXME: Do callbacks instead to avoid allocations.
/// Searches for importable items with the given name in the crate and its dependencies.
pub fn items_with_name(
    db: &RootDatabase,
    krate: Crate,
    name: NameToImport,
    assoc_item_search: AssocSearchMode,
) -> impl Iterator<Item = (ItemInNs, Complete)> {
    let _p = tracing::info_span!("items_with_name", name = name.text(), assoc_item_search = ?assoc_item_search, crate = ?krate.display_name(db).map(|name| name.to_string()))
        .entered();

    let prefix = matches!(name, NameToImport::Prefix(..));
    let (local_query, external_query) = match name {
        NameToImport::Prefix(exact_name, case_sensitive)
        | NameToImport::Exact(exact_name, case_sensitive) => {
            let mut local_query = symbol_index::Query::new(exact_name.clone());
            local_query.assoc_search_mode(assoc_item_search);
            let mut external_query =
                import_map::Query::new(exact_name).assoc_search_mode(assoc_item_search);
            if prefix {
                local_query.prefix();
                external_query = external_query.prefix();
            } else {
                local_query.exact();
                external_query = external_query.exact();
            }
            if case_sensitive {
                local_query.case_sensitive();
                external_query = external_query.case_sensitive();
            }
            (local_query, external_query)
        }
        NameToImport::Fuzzy(fuzzy_search_string, case_sensitive) => {
            let mut local_query = symbol_index::Query::new(fuzzy_search_string.clone());
            local_query.fuzzy();
            local_query.assoc_search_mode(assoc_item_search);

            let mut external_query = import_map::Query::new(fuzzy_search_string)
                .fuzzy()
                .assoc_search_mode(assoc_item_search);

            if case_sensitive {
                local_query.case_sensitive();
                external_query = external_query.case_sensitive();
            }

            (local_query, external_query)
        }
    };

    find_items(db, krate, local_query, external_query)
}

/// Searches for importable items with the given name in the crate and its dependencies.
pub fn items_with_name_in_module<T>(
    db: &RootDatabase,
    module: Module,
    name: NameToImport,
    assoc_item_search: AssocSearchMode,
    mut cb: impl FnMut(ItemInNs) -> ControlFlow<T>,
) -> Option<T> {
    let _p = tracing::info_span!("items_with_name_in", name = name.text(), assoc_item_search = ?assoc_item_search, ?module)
        .entered();

    let prefix = matches!(name, NameToImport::Prefix(..));
    let local_query = match name {
        NameToImport::Prefix(exact_name, case_sensitive)
        | NameToImport::Exact(exact_name, case_sensitive) => {
            let mut local_query = symbol_index::Query::new(exact_name);
            local_query.assoc_search_mode(assoc_item_search);
            if prefix {
                local_query.prefix();
            } else {
                local_query.exact();
            }
            if case_sensitive {
                local_query.case_sensitive();
            }
            local_query
        }
        NameToImport::Fuzzy(fuzzy_search_string, case_sensitive) => {
            let mut local_query = symbol_index::Query::new(fuzzy_search_string);
            local_query.fuzzy();
            local_query.assoc_search_mode(assoc_item_search);

            if case_sensitive {
                local_query.case_sensitive();
            }

            local_query
        }
    };
    local_query.search(&[db.module_symbols(module)], |local_candidate| {
        cb(match local_candidate.def {
            hir::ModuleDef::Macro(macro_def) => ItemInNs::Macros(macro_def),
            def => ItemInNs::from(def),
        })
    })
}

fn find_items(
    db: &RootDatabase,
    krate: Crate,
    local_query: symbol_index::Query,
    external_query: import_map::Query,
) -> impl Iterator<Item = (ItemInNs, Complete)> {
    let _p = tracing::info_span!("find_items").entered();

    // NOTE: `external_query` includes `assoc_item_search`, so we don't need to
    // filter on our own.
    let external_importables = krate.query_external_importables(db, external_query).map(
        |(external_importable, do_not_complete)| {
            let external_importable = match external_importable {
                Either::Left(module_def) => ItemInNs::from(module_def),
                Either::Right(macro_def) => ItemInNs::from(macro_def),
            };
            (external_importable, do_not_complete)
        },
    );

    // Query the local crate using the symbol index.
    let mut local_results = Vec::new();
    local_query.search(&symbol_index::crate_symbols(db, krate), |local_candidate| {
        let def = match local_candidate.def {
            hir::ModuleDef::Macro(macro_def) => ItemInNs::Macros(macro_def),
            def => ItemInNs::from(def),
        };
        local_results.push((def, local_candidate.do_not_complete));
        ControlFlow::<()>::Continue(())
    });
    local_results.into_iter().chain(external_importables)
}
