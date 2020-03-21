/// This declares a list of types which can be allocated by `Arena`.
///
/// The `few` modifier will cause allocation to use the shared arena and recording the destructor.
/// This is faster and more memory efficient if there's only a few allocations of the type.
/// Leaving `few` out will cause the type to get its own dedicated `TypedArena` which is
/// faster and more memory efficient if there is lots of allocations.
///
/// Specifying the `decode` modifier will add decode impls for &T and &[T] where T is the type
/// listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            [] layouts: rustc::ty::layout::LayoutDetails,
            [] generics: rustc::ty::Generics,
            [] trait_def: rustc::ty::TraitDef,
            [] adt_def: rustc::ty::AdtDef,
            [] steal_mir: rustc::ty::steal::Steal<rustc::mir::BodyAndCache<$tcx>>,
            [] mir: rustc::mir::BodyAndCache<$tcx>,
            [] steal_promoted: rustc::ty::steal::Steal<
                rustc_index::vec::IndexVec<
                    rustc::mir::Promoted,
                    rustc::mir::BodyAndCache<$tcx>
                >
            >,
            [] promoted: rustc_index::vec::IndexVec<
                rustc::mir::Promoted,
                rustc::mir::BodyAndCache<$tcx>
            >,
            [decode] tables: rustc::ty::TypeckTables<$tcx>,
            [decode] borrowck_result: rustc::mir::BorrowCheckResult<$tcx>,
            [] const_allocs: rustc::mir::interpret::Allocation,
            [] vtable_method: Option<(
                rustc_hir::def_id::DefId,
                rustc::ty::subst::SubstsRef<$tcx>
            )>,
            [few, decode] mir_keys: rustc_hir::def_id::DefIdSet,
            [decode] specialization_graph: rustc::traits::specialization_graph::Graph,
            [] region_scope_tree: rustc::middle::region::ScopeTree,
            [] item_local_set: rustc_hir::ItemLocalSet,
            [decode] mir_const_qualif: rustc_index::bit_set::BitSet<rustc::mir::Local>,
            [] trait_impls_of: rustc::ty::trait_def::TraitImpls,
            [] associated_items: rustc::ty::AssociatedItems,
            [] dropck_outlives:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        rustc::traits::query::DropckOutlivesResult<'tcx>
                    >
                >,
            [] normalize_projection_ty:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        rustc::traits::query::NormalizationResult<'tcx>
                    >
                >,
            [] implied_outlives_bounds:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx,
                        Vec<rustc::traits::query::OutlivesBound<'tcx>>
                    >
                >,
            [] type_op_subtype:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, ()>
                >,
            [] type_op_normalize_poly_fn_sig:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::PolyFnSig<'tcx>>
                >,
            [] type_op_normalize_fn_sig:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::FnSig<'tcx>>
                >,
            [] type_op_normalize_predicate:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::Predicate<'tcx>>
                >,
            [] type_op_normalize_ty:
                rustc::infer::canonical::Canonical<'tcx,
                    rustc::infer::canonical::QueryResponse<'tcx, rustc::ty::Ty<'tcx>>
                >,
            [few] crate_inherent_impls: rustc::ty::CrateInherentImpls,
            [few] upstream_monomorphizations:
                rustc_hir::def_id::DefIdMap<
                    rustc_data_structures::fx::FxHashMap<
                        rustc::ty::subst::SubstsRef<'tcx>,
                        rustc_hir::def_id::CrateNum
                    >
                >,
            [few] diagnostic_items: rustc_data_structures::fx::FxHashMap<
                rustc_span::symbol::Symbol,
                rustc_hir::def_id::DefId,
            >,
            [few] resolve_lifetimes: rustc::middle::resolve_lifetime::ResolveLifetimes,
            [few] lint_levels: rustc::lint::LintLevelMap,
            [few] stability_index: rustc::middle::stability::Index<'tcx>,
            [few] features: rustc_feature::Features,
            [few] all_traits: Vec<rustc_hir::def_id::DefId>,
            [few] privacy_access_levels: rustc::middle::privacy::AccessLevels,
            [few] target_features_whitelist: rustc_data_structures::fx::FxHashMap<
                String,
                Option<rustc_span::symbol::Symbol>
            >,
            [few] wasm_import_module_map: rustc_data_structures::fx::FxHashMap<
                rustc_hir::def_id::DefId,
                String
            >,
            [few] get_lib_features: rustc::middle::lib_features::LibFeatures,
            [few] defined_lib_features: rustc::middle::lang_items::LanguageItems,
            [few] visible_parent_map: rustc_hir::def_id::DefIdMap<rustc_hir::def_id::DefId>,
            [few] foreign_module: rustc::middle::cstore::ForeignModule,
            [few] foreign_modules: Vec<rustc::middle::cstore::ForeignModule>,
            [few] reachable_non_generics: rustc_hir::def_id::DefIdMap<
                rustc::middle::exported_symbols::SymbolExportLevel
            >,
            [few] crate_variances: rustc::ty::CrateVariancesMap<'tcx>,
            [few] inferred_outlives_crate: rustc::ty::CratePredicatesMap<'tcx>,
            [] upvars: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>,

            // Interned types
            [] tys: rustc::ty::TyS<$tcx>,

            // HIR query types
            [few] indexed_hir: rustc::hir::map::IndexedHir<$tcx>,
            [few] hir_definitions: rustc_hir::definitions::Definitions,
            [] hir_owner: rustc::hir::Owner<$tcx>,
            [] hir_owner_nodes: rustc::hir::OwnerNodes<$tcx>,
        ], $tcx);
    )
}

arena_types!(arena::declare_arena, [], 'tcx);
