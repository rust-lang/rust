//! Applies changes to the IDE state transactionally.

use std::sync::Arc;

use base_db::{
    salsa::{Database, Durability, SweepStrategy},
    Change, SourceRootId,
};
use profile::{memory_usage, Bytes};
use rustc_hash::FxHashSet;

use crate::{symbol_index::SymbolsDatabase, RootDatabase};

impl RootDatabase {
    pub fn request_cancellation(&mut self) {
        let _p = profile::span("RootDatabase::request_cancellation");
        self.salsa_runtime_mut().synthetic_write(Durability::LOW);
    }

    pub fn apply_change(&mut self, change: Change) {
        let _p = profile::span("RootDatabase::apply_change");
        self.request_cancellation();
        log::info!("apply_change {:?}", change);
        if let Some(roots) = &change.roots {
            let mut local_roots = FxHashSet::default();
            let mut library_roots = FxHashSet::default();
            for (idx, root) in roots.iter().enumerate() {
                let root_id = SourceRootId(idx as u32);
                if root.is_library {
                    library_roots.insert(root_id);
                } else {
                    local_roots.insert(root_id);
                }
            }
            self.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);
            self.set_library_roots_with_durability(Arc::new(library_roots), Durability::HIGH);
        }
        change.apply(self);
    }

    pub fn collect_garbage(&mut self) {
        if cfg!(target_arch = "wasm32") {
            return;
        }

        let _p = profile::span("RootDatabase::collect_garbage");

        let sweep = SweepStrategy::default().discard_values().sweep_all_revisions();

        base_db::ParseQuery.in_db(self).sweep(sweep);
        hir::db::ParseMacroExpansionQuery.in_db(self).sweep(sweep);

        // Macros do take significant space, but less then the syntax trees
        // self.query(hir::db::MacroDefQuery).sweep(sweep);
        // self.query(hir::db::MacroArgTextQuery).sweep(sweep);
        // self.query(hir::db::MacroExpandQuery).sweep(sweep);

        hir::db::AstIdMapQuery.in_db(self).sweep(sweep);

        hir::db::BodyWithSourceMapQuery.in_db(self).sweep(sweep);

        hir::db::ExprScopesQuery.in_db(self).sweep(sweep);
        hir::db::InferQueryQuery.in_db(self).sweep(sweep);
        hir::db::BodyQuery.in_db(self).sweep(sweep);
    }

    // Feature: Memory Usage
    //
    // Clears rust-analyzer's internal database and prints memory usage statistics.
    //
    // |===
    // | Editor  | Action Name
    //
    // | VS Code | **Rust Analyzer: Memory Usage (Clears Database)**
    // |===
    // image::https://user-images.githubusercontent.com/48062697/113065592-08559f00-91b1-11eb-8c96-64b88068ec02.gif[]
    pub fn per_query_memory_usage(&mut self) -> Vec<(String, Bytes)> {
        let mut acc: Vec<(String, Bytes)> = vec![];
        let sweep = SweepStrategy::default().discard_values().sweep_all_revisions();
        macro_rules! sweep_each_query {
            ($($q:path)*) => {$(
                let before = memory_usage().allocated;
                $q.in_db(self).sweep(sweep);
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?}", q);
                acc.push((name, before - after));

                let before = memory_usage().allocated;
                $q.in_db(self).sweep(sweep.discard_everything());
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?} (deps)", q);
                acc.push((name, before - after));

                let before = memory_usage().allocated;
                $q.in_db(self).purge();
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?} (purge)", q);
                acc.push((name, before - after));
            )*}
        }
        sweep_each_query![
            // SourceDatabase
            base_db::ParseQuery
            base_db::CrateGraphQuery

            // SourceDatabaseExt
            base_db::FileTextQuery
            base_db::FileSourceRootQuery
            base_db::SourceRootQuery
            base_db::SourceRootCratesQuery

            // AstDatabase
            hir::db::AstIdMapQuery
            hir::db::MacroArgTextQuery
            hir::db::MacroDefQuery
            hir::db::ParseMacroExpansionQuery
            hir::db::MacroExpandQuery
            hir::db::HygieneFrameQuery

            // DefDatabase
            hir::db::FileItemTreeQuery
            hir::db::BlockDefMapQuery
            hir::db::CrateDefMapQueryQuery
            hir::db::FieldsAttrsQuery
            hir::db::VariantsAttrsQuery
            hir::db::FieldsAttrsSourceMapQuery
            hir::db::VariantsAttrsSourceMapQuery
            hir::db::StructDataQuery
            hir::db::UnionDataQuery
            hir::db::EnumDataQuery
            hir::db::ImplDataQuery
            hir::db::TraitDataQuery
            hir::db::TypeAliasDataQuery
            hir::db::FunctionDataQuery
            hir::db::ConstDataQuery
            hir::db::StaticDataQuery
            hir::db::BodyWithSourceMapQuery
            hir::db::BodyQuery
            hir::db::ExprScopesQuery
            hir::db::GenericParamsQuery
            hir::db::AttrsQuery
            hir::db::CrateLangItemsQuery
            hir::db::LangItemQuery
            hir::db::ImportMapQuery

            // HirDatabase
            hir::db::InferQueryQuery
            hir::db::TyQuery
            hir::db::ValueTyQuery
            hir::db::ImplSelfTyQuery
            hir::db::ImplTraitQuery
            hir::db::FieldTypesQuery
            hir::db::CallableItemSignatureQuery
            hir::db::GenericPredicatesForParamQuery
            hir::db::GenericPredicatesQuery
            hir::db::GenericDefaultsQuery
            hir::db::InherentImplsInCrateQuery
            hir::db::TraitEnvironmentQuery
            hir::db::TraitImplsInCrateQuery
            hir::db::TraitImplsInDepsQuery
            hir::db::AssociatedTyDataQuery
            hir::db::AssociatedTyDataQuery
            hir::db::TraitDatumQuery
            hir::db::StructDatumQuery
            hir::db::ImplDatumQuery
            hir::db::FnDefDatumQuery
            hir::db::ReturnTypeImplTraitsQuery
            hir::db::InternCallableDefQuery
            hir::db::InternTypeParamIdQuery
            hir::db::InternImplTraitIdQuery
            hir::db::InternClosureQuery
            hir::db::AssociatedTyValueQuery
            hir::db::TraitSolveQueryQuery

            // SymbolsDatabase
            crate::symbol_index::FileSymbolsQuery
            crate::symbol_index::LibrarySymbolsQuery
            crate::symbol_index::LocalRootsQuery
            crate::symbol_index::LibraryRootsQuery

            // LineIndexDatabase
            crate::LineIndexQuery
        ];

        // To collect interned data, we need to bump the revision counter by performing a synthetic
        // write.
        // We do this after collecting the non-interned queries to correctly attribute memory used
        // by interned data.
        self.salsa_runtime_mut().synthetic_write(Durability::HIGH);

        sweep_each_query![
            // AstDatabase
            hir::db::InternMacroQuery

            // InternDatabase
            hir::db::InternFunctionQuery
            hir::db::InternStructQuery
            hir::db::InternUnionQuery
            hir::db::InternEnumQuery
            hir::db::InternConstQuery
            hir::db::InternStaticQuery
            hir::db::InternTraitQuery
            hir::db::InternTypeAliasQuery
            hir::db::InternImplQuery

            // HirDatabase
            hir::db::InternTypeParamIdQuery
        ];

        acc.sort_by_key(|it| std::cmp::Reverse(it.1));
        acc
    }
}
