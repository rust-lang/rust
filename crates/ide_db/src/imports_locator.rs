//! This module contains an import search funcionality that is provided to the assists module.
//! Later, this should be moved away to a separate crate that is accessible from the assists module.

use hir::{import_map, Crate, MacroDef, ModuleDef, Semantics};
use syntax::{ast, AstNode, SyntaxKind::NAME};

use crate::{
    defs::{Definition, NameClass},
    symbol_index::{self, FileSymbol},
    RootDatabase,
};
use either::Either;
use rustc_hash::FxHashSet;

pub fn find_exact_imports<'a>(
    sema: &Semantics<'a, RootDatabase>,
    krate: Crate,
    name_to_import: &str,
) -> impl Iterator<Item = Either<ModuleDef, MacroDef>> {
    let _p = profile::span("find_exact_imports");
    find_imports(
        sema,
        krate,
        {
            let mut local_query = symbol_index::Query::new(name_to_import.to_string());
            local_query.exact();
            local_query.limit(40);
            local_query
        },
        import_map::Query::new(name_to_import).anchor_end().case_sensitive().limit(40),
    )
}

pub fn find_similar_imports<'a>(
    sema: &Semantics<'a, RootDatabase>,
    krate: Crate,
    name_to_import: &str,
    ignore_modules: bool,
) -> impl Iterator<Item = Either<ModuleDef, MacroDef>> {
    let _p = profile::span("find_similar_imports");

    let mut external_query = import_map::Query::new(name_to_import);
    if ignore_modules {
        external_query = external_query.exclude_import_kind(import_map::ImportKind::Module);
    }

    find_imports(sema, krate, symbol_index::Query::new(name_to_import.to_string()), external_query)
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
