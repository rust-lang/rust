#![allow(rustc::usage_of_ty_tykind)]

/// This higher-order macro declares a list of types which can be allocated by `Arena`.
///
/// Specifying the `decode` modifier will add decode impls for `&T` and `&[T]` where `T` is the type
/// listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path) => (
        $macro!([
            [] layout: rustc_target::abi::LayoutS,
            [] fn_abi: rustc_target::abi::call::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
            // AdtDef are interned and compared by address
            [decode] adt_def: rustc_middle::ty::AdtDefData,
            [] steal_thir: rustc_data_structures::steal::Steal<rustc_middle::thir::Thir<'tcx>>,
            [] steal_mir: rustc_data_structures::steal::Steal<rustc_middle::mir::Body<'tcx>>,
            [decode] mir: rustc_middle::mir::Body<'tcx>,
            [] steal_promoted:
                rustc_data_structures::steal::Steal<
                    rustc_index::IndexVec<
                        rustc_middle::mir::Promoted,
                        rustc_middle::mir::Body<'tcx>
                    >
                >,
            [decode] promoted:
                rustc_index::IndexVec<
                    rustc_middle::mir::Promoted,
                    rustc_middle::mir::Body<'tcx>
                >,
            [decode] closure_debuginfo:
                rustc_index::IndexVec<
                    rustc_target::abi::FieldIdx,
                    rustc_span::symbol::Symbol,
                >,
            [decode] typeck_results: rustc_middle::ty::TypeckResults<'tcx>,
            [decode] borrowck_result:
                rustc_middle::mir::BorrowCheckResult<'tcx>,
            [] resolver: rustc_data_structures::steal::Steal<(
                rustc_middle::ty::ResolverAstLowering,
                rustc_data_structures::sync::Lrc<rustc_ast::Crate>,
            )>,
            [] output_filenames: std::sync::Arc<rustc_session::config::OutputFilenames>,
            [] metadata_loader: rustc_data_structures::steal::Steal<Box<rustc_session::cstore::MetadataLoaderDyn>>,
            [] crate_for_resolver: rustc_data_structures::steal::Steal<(rustc_ast::Crate, rustc_ast::AttrVec)>,
            [] resolutions: rustc_middle::ty::ResolverGlobalCtxt,
            [decode] unsafety_check_result: rustc_middle::mir::UnsafetyCheckResult,
            [decode] code_region: rustc_middle::mir::coverage::CodeRegion,
            [] const_allocs: rustc_middle::mir::interpret::Allocation,
            [] region_scope_tree: rustc_middle::middle::region::ScopeTree,
            // Required for the incremental on-disk cache
            [] mir_keys: rustc_hir::def_id::DefIdSet,
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
            [] dtorck_constraint: rustc_middle::traits::query::DropckConstraint<'tcx>,
            [] candidate_step: rustc_middle::traits::query::CandidateStep<'tcx>,
            [] autoderef_bad_ty: rustc_middle::traits::query::MethodAutoderefBadTy<'tcx>,
            [] query_region_constraints: rustc_middle::infer::canonical::QueryRegionConstraints<'tcx>,
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
            [] type_op_normalize_clause:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Clause<'tcx>>
                >,
            [] type_op_normalize_ty:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Ty<'tcx>>
                >,
            [] all_traits: Vec<rustc_hir::def_id::DefId>,
            [] effective_visibilities: rustc_middle::middle::privacy::EffectiveVisibilities,
            [] foreign_module: rustc_session::cstore::ForeignModule,
            [] foreign_modules: Vec<rustc_session::cstore::ForeignModule>,
            [] upvars_mentioned: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>,
            [] object_safety_violations: rustc_middle::traits::ObjectSafetyViolation,
            [] codegen_unit: rustc_middle::mir::mono::CodegenUnit<'tcx>,
            [decode] attribute: rustc_ast::Attribute,
            [] name_set: rustc_data_structures::unord::UnordSet<rustc_span::symbol::Symbol>,
            [] ordered_name_set: rustc_data_structures::fx::FxIndexSet<rustc_span::symbol::Symbol>,
            [] hir_id_set: rustc_hir::HirIdSet,

            // Interned types
            [] tys: rustc_type_ir::WithCachedTypeInfo<rustc_middle::ty::TyKind<'tcx>>,
            [] predicates: rustc_type_ir::WithCachedTypeInfo<rustc_middle::ty::PredicateKind<'tcx>>,
            [] consts: rustc_middle::ty::ConstData<'tcx>,

            // Note that this deliberately duplicates items in the `rustc_hir::arena`,
            // since we need to allocate this type on both the `rustc_hir` arena
            // (during lowering) and the `librustc_middle` arena (for decoding MIR)
            [decode] asm_template: rustc_ast::InlineAsmTemplatePiece,
            [decode] used_trait_imports: rustc_data_structures::unord::UnordSet<rustc_hir::def_id::LocalDefId>,
            [decode] registered_tools: rustc_middle::ty::RegisteredTools,
            [decode] is_late_bound_map: rustc_data_structures::fx::FxIndexSet<rustc_hir::ItemLocalId>,
            [decode] impl_source: rustc_middle::traits::ImplSource<'tcx, ()>,

            [] dep_kind: rustc_middle::dep_graph::DepKindStruct<'tcx>,

            [decode] trait_impl_trait_tys:
                rustc_data_structures::fx::FxHashMap<
                    rustc_hir::def_id::DefId,
                    rustc_middle::ty::EarlyBinder<rustc_middle::ty::Ty<'tcx>>
                >,
            [] bit_set_u32: rustc_index::bit_set::BitSet<u32>,
            [] external_constraints: rustc_middle::traits::solve::ExternalConstraintsData<'tcx>,
            [] predefined_opaques_in_body: rustc_middle::traits::solve::PredefinedOpaquesData<'tcx>,
            [decode] doc_link_resolutions: rustc_hir::def::DocLinkResMap,
            [] closure_kind_origin: (rustc_span::Span, rustc_middle::hir::place::Place<'tcx>),
            [] stripped_cfg_items: rustc_ast::expand::StrippedCfgItem,
            [] mod_child: rustc_middle::metadata::ModChild,
        ]);
    )
}

arena_types!(rustc_arena::declare_arena);
