//! This module contains an import search functionality that is provided to the assists module.
//! Later, this should be moved away to a separate crate that is accessible from the assists module.

use hir::{
    import_map::{self, ImportKind},
    AsAssocItem, Crate, MacroDef, ModuleDef, Semantics,
};
use syntax::{ast, AstNode, SyntaxKind::NAME};

use crate::{
    defs::{Definition, NameClass},
    symbol_index::{self, FileSymbol},
    RootDatabase,
};
use either::Either;
use rustc_hash::FxHashSet;

pub(crate) const DEFAULT_QUERY_SEARCH_LIMIT: usize = 40;

pub fn find_exact_imports(
    sema: &Semantics<'_, RootDatabase>,
    krate: Crate,
    name_to_import: String,
) -> Box<dyn Iterator<Item = Either<ModuleDef, MacroDef>>> {
    let _p = profile::span("find_exact_imports");
    Box::new(find_imports(
        sema,
        krate,
        {
            let mut local_query = symbol_index::Query::new(name_to_import.clone());
            local_query.exact();
            local_query.limit(DEFAULT_QUERY_SEARCH_LIMIT);
            local_query
        },
        import_map::Query::new(name_to_import)
            .limit(DEFAULT_QUERY_SEARCH_LIMIT)
            .name_only()
            .search_mode(import_map::SearchMode::Equals)
            .case_sensitive(),
    ))
}

#[derive(Debug)]
pub enum AssocItemSearch {
    Include,
    Exclude,
    AssocItemsOnly,
}

pub fn find_similar_imports<'a>(
    sema: &'a Semantics<'a, RootDatabase>,
    krate: Crate,
    fuzzy_search_string: String,
    assoc_item_search: AssocItemSearch,
    limit: Option<usize>,
) -> Box<dyn Iterator<Item = Either<ModuleDef, MacroDef>> + 'a> {
    let _p = profile::span("find_similar_imports");

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

    let mut local_query = symbol_index::Query::new(fuzzy_search_string);

    if let Some(limit) = limit {
        external_query = external_query.limit(limit);
        local_query.limit(limit);
    }

    Box::new(find_imports(sema, krate, local_query, external_query).filter(
        move |import_candidate| match assoc_item_search {
            AssocItemSearch::Include => true,
            AssocItemSearch::Exclude => !is_assoc_item(import_candidate, sema.db),
            AssocItemSearch::AssocItemsOnly => is_assoc_item(import_candidate, sema.db),
        },
    ))
}

fn is_assoc_item(import_candidate: &Either<ModuleDef, MacroDef>, db: &RootDatabase) -> bool {
    match import_candidate {
        Either::Left(ModuleDef::Function(function)) => function.as_assoc_item(db).is_some(),
        Either::Left(ModuleDef::Const(const_)) => const_.as_assoc_item(db).is_some(),
        Either::Left(ModuleDef::TypeAlias(type_alias)) => type_alias.as_assoc_item(db).is_some(),
        _ => false,
    }
}

fn find_imports<'a>(
    sema: &Semantics<'a, RootDatabase>,
    krate: Crate,
    local_query: symbol_index::Query,
    external_query: import_map::Query,
) -> impl Iterator<Item = Either<ModuleDef, MacroDef>> {
    let _p = profile::span("find_similar_imports");
    let db = sema.db;

    // Query dependencies first.
    let mut candidates: FxHashSet<_> =
        krate.query_external_importables(db, external_query).collect();

    // Query the local crate using the symbol index.
    let local_results = symbol_index::crate_symbols(db, krate.into(), local_query);

    candidates.extend(
        local_results
            .into_iter()
            .filter_map(|import_candidate| get_name_definition(sema, &import_candidate))
            .filter_map(|name_definition_to_import| match name_definition_to_import {
                Definition::ModuleDef(module_def) => Some(Either::Left(module_def)),
                Definition::Macro(macro_def) => Some(Either::Right(macro_def)),
                _ => None,
            }),
    );

    candidates.into_iter()
}

fn get_name_definition<'a>(
    sema: &Semantics<'a, RootDatabase>,
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
