//! Applies changes to the IDE state transactionally.

use base_db::SourceRootId;
use profile::Bytes;
use rustc_hash::FxHashSet;
use salsa::{Database as _, Durability};
use triomphe::Arc;

use crate::{ChangeWithProcMacros, RootDatabase, symbol_index::SymbolsDatabase};

impl RootDatabase {
    pub fn request_cancellation(&mut self) {
        let _p = tracing::info_span!("RootDatabase::request_cancellation").entered();
        self.synthetic_write(Durability::LOW);
    }

    pub fn apply_change(&mut self, change: ChangeWithProcMacros) {
        let _p = tracing::info_span!("RootDatabase::apply_change").entered();
        self.request_cancellation();
        tracing::trace!("apply_change {:?}", change);
        if let Some(roots) = &change.source_change.roots {
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
            self.set_local_roots_with_durability(Arc::new(local_roots), Durability::MEDIUM);
            self.set_library_roots_with_durability(Arc::new(library_roots), Durability::MEDIUM);
        }
        change.apply(self);
    }

    // Feature: Memory Usage
    //
    // Clears rust-analyzer's internal database and prints memory usage statistics.
    //
    // | Editor  | Action Name |
    // |---------|-------------|
    // | VS Code | **rust-analyzer: Memory Usage (Clears Database)**

    // ![Memory Usage](https://user-images.githubusercontent.com/48062697/113065592-08559f00-91b1-11eb-8c96-64b88068ec02.gif)
    pub fn per_query_memory_usage(&mut self) -> Vec<(String, Bytes, usize)> {
        let mut acc: Vec<(String, Bytes, usize)> = vec![];

        // fn collect_query_count<'q, Q>(table: &QueryTable<'q, Q>) -> usize
        // where
        //     QueryTable<'q, Q>: DebugQueryTable,
        //     Q: Query,
        //     <Q as Query>::Storage: 'q,
        // {
        //     struct EntryCounter(usize);
        //     impl<K, V> FromIterator<TableEntry<K, V>> for EntryCounter {
        //         fn from_iter<T>(iter: T) -> EntryCounter
        //         where
        //             T: IntoIterator<Item = TableEntry<K, V>>,
        //         {
        //             EntryCounter(iter.into_iter().count())
        //         }
        //     }
        //     table.entries::<EntryCounter>().0
        // }

        macro_rules! purge_each_query {
            ($($q:path)*) => {$(
                let before = memory_usage().allocated;
                let table = $q.in_db(self);
                let count = collect_query_count(&table);
                table.purge();
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?}", q);
                acc.push((name, before - after, count));
            )*}
        }
        purge_each_query![
            // // SymbolsDatabase
            // crate::symbol_index::ModuleSymbolsQuery
            // crate::symbol_index::LibrarySymbolsQuery
            // crate::symbol_index::LocalRootsQuery
            // crate::symbol_index::LibraryRootsQuery
            // // HirDatabase
            // hir::db::AdtDatumQuery
            // hir::db::AdtVarianceQuery
            // hir::db::AssociatedTyDataQuery
            // hir::db::AssociatedTyValueQuery
            // hir::db::BorrowckQuery
            // hir::db::CallableItemSignatureQuery
            // hir::db::ConstEvalDiscriminantQuery
            // hir::db::ConstEvalQuery
            // hir::db::ConstEvalStaticQuery
            // hir::db::ConstParamTyQuery
            // hir::db::DynCompatibilityOfTraitQuery
            // hir::db::FieldTypesQuery
            // hir::db::FnDefDatumQuery
            // hir::db::FnDefVarianceQuery
            // hir::db::GenericDefaultsQuery
            // hir::db::GenericPredicatesForParamQuery
            // hir::db::GenericPredicatesQuery
            // hir::db::GenericPredicatesWithoutParentQuery
            // hir::db::ImplDatumQuery
            // hir::db::ImplSelfTyQuery
            // hir::db::ImplTraitQuery
            // hir::db::IncoherentInherentImplCratesQuery
            // hir::db::InferQuery
            // hir::db::InherentImplsInBlockQuery
            // hir::db::InherentImplsInCrateQuery
            // hir::db::InternCallableDefQuery
            // hir::db::InternClosureQuery
            // hir::db::InternCoroutineQuery
            // hir::db::InternImplTraitIdQuery
            // hir::db::InternLifetimeParamIdQuery
            // hir::db::InternTypeOrConstParamIdQuery
            // hir::db::LayoutOfAdtQuery
            // hir::db::LayoutOfTyQuery
            // hir::db::LookupImplMethodQuery
            // hir::db::MirBodyForClosureQuery
            // hir::db::MirBodyQuery
            // hir::db::MonomorphizedMirBodyForClosureQuery
            // hir::db::MonomorphizedMirBodyQuery
            // hir::db::ProgramClausesForChalkEnvQuery
            // hir::db::ReturnTypeImplTraitsQuery
            // hir::db::TargetDataLayoutQuery
            // hir::db::TraitDatumQuery
            // hir::db::TraitEnvironmentQuery
            // hir::db::TraitImplsInBlockQuery
            // hir::db::TraitImplsInCrateQuery
            // hir::db::TraitImplsInDepsQuery
            // hir::db::TraitSolveQuery
            // hir::db::TyQuery
            // hir::db::TypeAliasImplTraitsQuery
            // hir::db::ValueTyQuery

            // // DefDatabase
            // hir::db::AttrsQuery
            // hir::db::BlockDefMapQuery
            // hir::db::BlockItemTreeQuery
            // hir::db::BlockItemTreeWithSourceMapQuery
            // hir::db::BodyQuery
            // hir::db::BodyWithSourceMapQuery
            // hir::db::ConstDataQuery
            // hir::db::ConstVisibilityQuery
            // hir::db::CrateDefMapQuery
            // hir::db::CrateLangItemsQuery
            // hir::db::CrateNotableTraitsQuery
            // hir::db::CrateSupportsNoStdQuery
            // hir::db::EnumDataQuery
            // hir::db::ExpandProcAttrMacrosQuery
            // hir::db::ExprScopesQuery
            // hir::db::ExternCrateDeclDataQuery
            // hir::db::FieldVisibilitiesQuery
            // hir::db::FieldsAttrsQuery
            // hir::db::FieldsAttrsSourceMapQuery
            // hir::db::FileItemTreeQuery
            // hir::db::FileItemTreeWithSourceMapQuery
            // hir::db::FunctionDataQuery
            // hir::db::FunctionVisibilityQuery
            // hir::db::GenericParamsQuery
            // hir::db::GenericParamsWithSourceMapQuery
            // hir::db::ImplItemsWithDiagnosticsQuery
            // hir::db::ImportMapQuery
            // hir::db::IncludeMacroInvocQuery
            // hir::db::InternAnonymousConstQuery
            // hir::db::InternBlockQuery
            // hir::db::InternConstQuery
            // hir::db::InternEnumQuery
            // hir::db::InternExternBlockQuery
            // hir::db::InternExternCrateQuery
            // hir::db::InternFunctionQuery
            // hir::db::InternImplQuery
            // hir::db::InternInTypeConstQuery
            // hir::db::InternMacro2Query
            // hir::db::InternMacroRulesQuery
            // hir::db::InternProcMacroQuery
            // hir::db::InternStaticQuery
            // hir::db::InternStructQuery
            // hir::db::InternTraitAliasQuery
            // hir::db::InternTraitQuery
            // hir::db::InternTypeAliasQuery
            // hir::db::InternUnionQuery
            // hir::db::InternUseQuery
            // hir::db::LangItemQuery
            // hir::db::Macro2DataQuery
            // hir::db::MacroDefQuery
            // hir::db::MacroRulesDataQuery
            // hir::db::NotableTraitsInDepsQuery
            // hir::db::ProcMacroDataQuery
            // hir::db::StaticDataQuery
            // hir::db::TraitAliasDataQuery
            // hir::db::TraitItemsWithDiagnosticsQuery
            // hir::db::TypeAliasDataQuery
            // hir::db::VariantDataWithDiagnosticsQuery

            // // InternDatabase
            // hir::db::InternFunctionQuery
            // hir::db::InternStructQuery
            // hir::db::InternUnionQuery
            // hir::db::InternEnumQuery
            // hir::db::InternConstQuery
            // hir::db::InternStaticQuery
            // hir::db::InternTraitQuery
            // hir::db::InternTraitAliasQuery
            // hir::db::InternTypeAliasQuery
            // hir::db::InternImplQuery
            // hir::db::InternExternBlockQuery
            // hir::db::InternBlockQuery
            // hir::db::InternMacro2Query
            // hir::db::InternProcMacroQuery
            // hir::db::InternMacroRulesQuery

            // // ExpandDatabase
            // hir::db::AstIdMapQuery
            // hir::db::DeclMacroExpanderQuery
            // hir::db::ExpandProcMacroQuery
            // hir::db::InternMacroCallQuery
            // hir::db::InternSyntaxContextQuery
            // hir::db::MacroArgQuery
            // hir::db::ParseMacroExpansionErrorQuery
            // hir::db::ParseMacroExpansionQuery
            // hir::db::ProcMacroSpanQuery
            // hir::db::ProcMacrosQuery
            // hir::db::RealSpanMapQuery

            // // LineIndexDatabase
            // crate::LineIndexQuery

            // // SourceDatabase
            // base_db::ParseQuery
            // base_db::ParseErrorsQuery
            // base_db::AllCratesQuery
            // base_db::InternUniqueCrateDataQuery
            // base_db::InternUniqueCrateDataLookupQuery
            // base_db::CrateDataQuery
            // base_db::ExtraCrateDataQuery
            // base_db::CrateCfgQuery
            // base_db::CrateEnvQuery
            // base_db::CrateWorkspaceDataQuery

            // // SourceDatabaseExt
            // base_db::FileTextQuery
            // base_db::CompressedFileTextQuery
            // base_db::FileSourceRootQuery
            // base_db::SourceRootQuery
            // base_db::SourceRootCratesQuery
        ];

        acc.sort_by_key(|it| std::cmp::Reverse(it.1));
        acc
    }
}
