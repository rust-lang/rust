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
            [] layouts: rustc_middle::ty::layout::Layout,
            [] generics: rustc_middle::ty::Generics,
            [] trait_def: rustc_middle::ty::TraitDef,
            [] adt_def: rustc_middle::ty::AdtDef,
            [] steal_mir: rustc_middle::ty::steal::Steal<rustc_middle::mir::BodyAndCache<$tcx>>,
            [] mir: rustc_middle::mir::BodyAndCache<$tcx>,
            [] steal_promoted: rustc_middle::ty::steal::Steal<
                rustc_index::vec::IndexVec<
                    rustc_middle::mir::Promoted,
                    rustc_middle::mir::BodyAndCache<$tcx>
                >
            >,
            [] promoted: rustc_index::vec::IndexVec<
                rustc_middle::mir::Promoted,
                rustc_middle::mir::BodyAndCache<$tcx>
            >,
            [decode] tables: rustc_middle::ty::TypeckTables<$tcx>,
            [decode] borrowck_result: rustc_middle::mir::BorrowCheckResult<$tcx>,
            [] const_allocs: rustc_middle::mir::interpret::Allocation,
            [] vtable_method: Option<(
                rustc_hir::def_id::DefId,
                rustc_middle::ty::subst::SubstsRef<$tcx>
            )>,
            [few, decode] mir_keys: rustc_hir::def_id::DefIdSet,
            [decode] specialization_graph: rustc_middle::traits::specialization_graph::Graph,
            [] region_scope_tree: rustc_middle::middle::region::ScopeTree,
            [] item_local_set: rustc_hir::ItemLocalSet,
            [decode] mir_const_qualif: rustc_index::bit_set::BitSet<rustc_middle::mir::Local>,
            [] trait_impls_of: rustc_middle::ty::trait_def::TraitImpls,
            [] associated_items: rustc_middle::ty::AssociatedItems,
            [] dropck_outlives:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        rustc_middle::traits::query::DropckOutlivesResult<'tcx>
                    >
                >,
            [] normalize_projection_ty:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        rustc_middle::traits::query::NormalizationResult<'tcx>
                    >
                >,
            [] implied_outlives_bounds:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        Vec<rustc_middle::traits::query::OutlivesBound<'tcx>>
                    >
                >,
            [] type_op_subtype:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, ()>
                >,
            [] type_op_normalize_poly_fn_sig:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::PolyFnSig<'tcx>>
                >,
            [] type_op_normalize_fn_sig:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::FnSig<'tcx>>
                >,
            [] type_op_normalize_predicate:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Predicate<'tcx>>
                >,
            [] type_op_normalize_ty:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Ty<'tcx>>
                >,
            [few] crate_inherent_impls: rustc_middle::ty::CrateInherentImpls,
            [few] upstream_monomorphizations:
                rustc_hir::def_id::DefIdMap<
                    rustc_data_structures::fx::FxHashMap<
                        rustc_middle::ty::subst::SubstsRef<'tcx>,
                        rustc_hir::def_id::CrateNum
                    >
                >,
            [few] diagnostic_items: rustc_data_structures::fx::FxHashMap<
                rustc_span::symbol::Symbol,
                rustc_hir::def_id::DefId,
            >,
            [few] resolve_lifetimes: rustc_middle::middle::resolve_lifetime::ResolveLifetimes,
            [few] lint_levels: rustc_middle::lint::LintLevelMap,
            [few] stability_index: rustc_middle::middle::stability::Index<'tcx>,
            [few] features: rustc_feature::Features,
            [few] all_traits: Vec<rustc_hir::def_id::DefId>,
            [few] privacy_access_levels: rustc_middle::middle::privacy::AccessLevels,
            [few] target_features_whitelist: rustc_data_structures::fx::FxHashMap<
                String,
                Option<rustc_span::symbol::Symbol>
            >,
            [few] wasm_import_module_map: rustc_data_structures::fx::FxHashMap<
                rustc_hir::def_id::DefId,
                String
            >,
            [few] get_lib_features: rustc_middle::middle::lib_features::LibFeatures,
            [few] defined_lib_features: rustc_middle::middle::lang_items::LanguageItems,
            [few] visible_parent_map: rustc_hir::def_id::DefIdMap<rustc_hir::def_id::DefId>,
            [few] foreign_module: rustc_middle::middle::cstore::ForeignModule,
            [few] foreign_modules: Vec<rustc_middle::middle::cstore::ForeignModule>,
            [few] reachable_non_generics: rustc_hir::def_id::DefIdMap<
                rustc_middle::middle::exported_symbols::SymbolExportLevel
            >,
            [few] crate_variances: rustc_middle::ty::CrateVariancesMap<'tcx>,
            [few] inferred_outlives_crate: rustc_middle::ty::CratePredicatesMap<'tcx>,
            [] upvars: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>,

            // Interned types
            [] tys: rustc_middle::ty::TyS<$tcx>,

            // HIR query types
            [few] indexed_hir: rustc_middle::hir::map::IndexedHir<$tcx>,
            [few] hir_definitions: rustc_hir::definitions::Definitions,
            [] hir_owner: rustc_middle::hir::Owner<$tcx>,
            [] hir_owner_nodes: rustc_middle::hir::OwnerNodes<$tcx>,
        ], $tcx);
    )
}

arena_types!(arena::declare_arena, [], 'tcx);
