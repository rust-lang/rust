//! This module has the functionality to search the project and its dependencies for a certain item,
//! by its name and a few criteria.
//! The main reason for this module to exist is the fact that project's items and dependencies' items
//! are located in different caches, with different APIs.
use either::Either;
use hir::{import_map, Crate, ItemInNs, Semantics};
use limit::Limit;

use crate::{imports::import_assets::NameToImport, symbol_index, RootDatabase};

/// A value to use, when uncertain which limit to pick.
pub static DEFAULT_QUERY_SEARCH_LIMIT: Limit = Limit::new(100);

pub use import_map::AssocSearchMode;

/// Searches for importable items with the given name in the crate and its dependencies.
pub fn items_with_name<'a>(
    sema: &'a Semantics<'_, RootDatabase>,
    krate: Crate,
    name: NameToImport,
    assoc_item_search: AssocSearchMode,
) -> impl Iterator<Item = ItemInNs> + 'a {
    let krate_name = krate.display_name(sema.db).map(|name| name.to_string());
    let _p = tracing::info_span!("items_with_name", name = name.text(), assoc_item_search = ?assoc_item_search, crate = ?krate_name)
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

    find_items(sema, krate, local_query, external_query)
}

fn find_items<'a>(
    sema: &'a Semantics<'_, RootDatabase>,
    krate: Crate,
    local_query: symbol_index::Query,
    external_query: import_map::Query,
) -> impl Iterator<Item = ItemInNs> + 'a {
    let _p = tracing::info_span!("find_items").entered();
    let db = sema.db;

    // NOTE: `external_query` includes `assoc_item_search`, so we don't need to
    // filter on our own.
    let external_importables =
        krate.query_external_importables(db, external_query).map(|external_importable| {
            match external_importable {
                Either::Left(module_def) => ItemInNs::from(module_def),
                Either::Right(macro_def) => ItemInNs::from(macro_def),
            }
        });

    // Query the local crate using the symbol index.
    let mut local_results = Vec::new();
    local_query.search(&symbol_index::crate_symbols(db, krate), |local_candidate| {
        local_results.push(match local_candidate.def {
            hir::ModuleDef::Macro(macro_def) => ItemInNs::Macros(macro_def),
            def => ItemInNs::from(def),
        })
    });
    local_results.into_iter().chain(external_importables)
}
