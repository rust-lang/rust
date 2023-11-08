//! This module has the functionality to search the project and its dependencies for a certain item,
//! by its name and a few criteria.
//! The main reason for this module to exist is the fact that project's items and dependencies' items
//! are located in different caches, with different APIs.
use either::Either;
use hir::{import_map, AsAssocItem, Crate, ItemInNs, Semantics};
use limit::Limit;

use crate::{imports::import_assets::NameToImport, symbol_index, RootDatabase};

/// A value to use, when uncertain which limit to pick.
pub static DEFAULT_QUERY_SEARCH_LIMIT: Limit = Limit::new(40);

pub use import_map::AssocSearchMode;

/// Searches for importable items with the given name in the crate and its dependencies.
pub fn items_with_name<'a>(
    sema: &'a Semantics<'_, RootDatabase>,
    krate: Crate,
    name: NameToImport,
    assoc_item_search: AssocSearchMode,
    limit: Option<usize>,
) -> impl Iterator<Item = ItemInNs> + 'a {
    let _p = profile::span("items_with_name").detail(|| {
        format!(
            "Name: {}, crate: {:?}, assoc items: {:?}, limit: {:?}",
            name.text(),
            assoc_item_search,
            krate.display_name(sema.db).map(|name| name.to_string()),
            limit,
        )
    });

    let prefix = matches!(name, NameToImport::Prefix(..));
    let (mut local_query, mut external_query) = match name {
        NameToImport::Prefix(exact_name, case_sensitive)
        | NameToImport::Exact(exact_name, case_sensitive) => {
            let mut local_query = symbol_index::Query::new(exact_name.clone());
            let mut external_query = import_map::Query::new(exact_name);
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

            let mut external_query = import_map::Query::new(fuzzy_search_string.clone())
                .fuzzy()
                .assoc_search_mode(assoc_item_search);

            if case_sensitive {
                local_query.case_sensitive();
                external_query = external_query.case_sensitive();
            }

            (local_query, external_query)
        }
    };

    if let Some(limit) = limit {
        external_query = external_query.limit(limit);
        local_query.limit(limit);
    }

    find_items(sema, krate, assoc_item_search, local_query, external_query)
}

fn find_items<'a>(
    sema: &'a Semantics<'_, RootDatabase>,
    krate: Crate,
    assoc_item_search: AssocSearchMode,
    local_query: symbol_index::Query,
    external_query: import_map::Query,
) -> impl Iterator<Item = ItemInNs> + 'a {
    let _p = profile::span("find_items");
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
    let local_results = local_query
        .search(&symbol_index::crate_symbols(db, krate))
        .into_iter()
        .filter(move |candidate| match assoc_item_search {
            AssocSearchMode::Include => true,
            AssocSearchMode::Exclude => candidate.def.as_assoc_item(db).is_none(),
            AssocSearchMode::AssocItemsOnly => candidate.def.as_assoc_item(db).is_some(),
        })
        .map(|local_candidate| match local_candidate.def {
            hir::ModuleDef::Macro(macro_def) => ItemInNs::Macros(macro_def),
            def => ItemInNs::from(def),
        });

    external_importables.chain(local_results)
}
