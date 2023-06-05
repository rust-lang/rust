//! Applies changes to the IDE state transactionally.

use base_db::{
    salsa::{Database, Durability},
    Change, SourceRootId,
};
use profile::{memory_usage, Bytes};
use rustc_hash::FxHashSet;
use triomphe::Arc;

use crate::{symbol_index::SymbolsDatabase, RootDatabase};

impl RootDatabase {
    pub fn request_cancellation(&mut self) {
        let _p = profile::span("RootDatabase::request_cancellation");
        self.salsa_runtime_mut().synthetic_write(Durability::LOW);
    }

    pub fn apply_change(&mut self, change: Change) {
        let _p = profile::span("RootDatabase::apply_change");
        self.request_cancellation();
        tracing::trace!("apply_change {:?}", change);
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

    // Feature: Memory Usage
    //
    // Clears rust-analyzer's internal database and prints memory usage statistics.
    //
    // |===
    // | Editor  | Action Name
    //
    // | VS Code | **rust-analyzer: Memory Usage (Clears Database)**
    // |===
    // image::https://user-images.githubusercontent.com/48062697/113065592-08559f00-91b1-11eb-8c96-64b88068ec02.gif[]
    pub fn per_query_memory_usage(&mut self) -> Vec<(String, Bytes)> {
        let mut acc: Vec<(String, Bytes)> = vec![];
        macro_rules! purge_each_query {
            ($($q:path)*) => {$(
                let before = memory_usage().allocated;
                $q.in_db(self).purge();
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?}", q);
                acc.push((name, before - after));
            )*}
        }
        purge_each_query![
            // SourceDatabase
            base_db::ParseQuery
            base_db::CrateGraphQuery
            base_db::ProcMacrosQuery

            // SourceDatabaseExt
            base_db::FileTextQuery
            base_db::FileSourceRootQuery
            base_db::SourceRootQuery
            base_db::SourceRootCratesQuery

            // ExpandDatabase
            hir::db::AstIdMapQuery
            hir::db::ParseMacroExpansionQuery
            hir::db::InternMacroCallQuery
            hir::db::MacroArgTextQuery
            hir::db::MacroDefQuery
            hir::db::MacroExpandQuery
            hir::db::ExpandProcMacroQuery
            hir::db::HygieneFrameQuery

            // DefDatabase
            hir::db::FileItemTreeQuery
            hir::db::CrateDefMapQueryQuery
            hir::db::BlockDefMapQuery
            hir::db::StructDataQuery
            hir::db::StructDataWithDiagnosticsQuery
            hir::db::UnionDataQuery
            hir::db::UnionDataWithDiagnosticsQuery
            hir::db::EnumDataQuery
            hir::db::EnumDataWithDiagnosticsQuery
            hir::db::ImplDataQuery
            hir::db::ImplDataWithDiagnosticsQuery
            hir::db::TraitDataQuery
            hir::db::TraitDataWithDiagnosticsQuery
            hir::db::TraitAliasDataQuery
            hir::db::TypeAliasDataQuery
            hir::db::FunctionDataQuery
            hir::db::ConstDataQuery
            hir::db::StaticDataQuery
            hir::db::Macro2DataQuery
            hir::db::MacroRulesDataQuery
            hir::db::ProcMacroDataQuery
            hir::db::BodyWithSourceMapQuery
            hir::db::BodyQuery
            hir::db::ExprScopesQuery
            hir::db::GenericParamsQuery
            hir::db::VariantsAttrsQuery
            hir::db::FieldsAttrsQuery
            hir::db::VariantsAttrsSourceMapQuery
            hir::db::FieldsAttrsSourceMapQuery
            hir::db::AttrsQuery
            hir::db::CrateLangItemsQuery
            hir::db::LangItemQuery
            hir::db::ImportMapQuery
            hir::db::FieldVisibilitiesQuery
            hir::db::FunctionVisibilityQuery
            hir::db::ConstVisibilityQuery
            hir::db::CrateSupportsNoStdQuery

            // HirDatabase
            hir::db::InferQueryQuery
            hir::db::MirBodyQuery
            hir::db::BorrowckQuery
            hir::db::TyQuery
            hir::db::ValueTyQuery
            hir::db::ImplSelfTyQuery
            hir::db::ConstParamTyQuery
            hir::db::ConstEvalQuery
            hir::db::ConstEvalDiscriminantQuery
            hir::db::ImplTraitQuery
            hir::db::FieldTypesQuery
            hir::db::LayoutOfAdtQuery
            hir::db::TargetDataLayoutQuery
            hir::db::CallableItemSignatureQuery
            hir::db::ReturnTypeImplTraitsQuery
            hir::db::GenericPredicatesForParamQuery
            hir::db::GenericPredicatesQuery
            hir::db::TraitEnvironmentQuery
            hir::db::GenericDefaultsQuery
            hir::db::InherentImplsInCrateQuery
            hir::db::InherentImplsInBlockQuery
            hir::db::IncoherentInherentImplCratesQuery
            hir::db::TraitImplsInCrateQuery
            hir::db::TraitImplsInBlockQuery
            hir::db::TraitImplsInDepsQuery
            hir::db::InternCallableDefQuery
            hir::db::InternLifetimeParamIdQuery
            hir::db::InternImplTraitIdQuery
            hir::db::InternTypeOrConstParamIdQuery
            hir::db::InternClosureQuery
            hir::db::InternGeneratorQuery
            hir::db::AssociatedTyDataQuery
            hir::db::TraitDatumQuery
            hir::db::StructDatumQuery
            hir::db::ImplDatumQuery
            hir::db::FnDefDatumQuery
            hir::db::FnDefVarianceQuery
            hir::db::AdtVarianceQuery
            hir::db::AssociatedTyValueQuery
            hir::db::TraitSolveQueryQuery
            hir::db::ProgramClausesForChalkEnvQuery

            // SymbolsDatabase
            crate::symbol_index::ModuleSymbolsQuery
            crate::symbol_index::LibrarySymbolsQuery
            crate::symbol_index::LocalRootsQuery
            crate::symbol_index::LibraryRootsQuery

            // LineIndexDatabase
            crate::LineIndexQuery

            // InternDatabase
            hir::db::InternFunctionQuery
            hir::db::InternStructQuery
            hir::db::InternUnionQuery
            hir::db::InternEnumQuery
            hir::db::InternConstQuery
            hir::db::InternStaticQuery
            hir::db::InternTraitQuery
            hir::db::InternTraitAliasQuery
            hir::db::InternTypeAliasQuery
            hir::db::InternImplQuery
            hir::db::InternExternBlockQuery
            hir::db::InternBlockQuery
            hir::db::InternMacro2Query
            hir::db::InternProcMacroQuery
            hir::db::InternMacroRulesQuery
        ];

        acc.sort_by_key(|it| std::cmp::Reverse(it.1));
        acc
    }
}
