/// This higher-order macro declares a list of types which can be allocated by `Arena`.
///
/// Specifying the `decode` modifier will add decode impls for `&T` and `&[T]` where `T` is the type
/// listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path) => (
        $macro!([
            [] layout: rustc_target::abi::Layout,
            [] fn_abi: rustc_target::abi::call::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
            // AdtDef are interned and compared by address
            [] adt_def: rustc_middle::ty::AdtDef,
            [] steal_thir: rustc_data_structures::steal::Steal<rustc_middle::thir::Thir<'tcx>>,
            [] steal_mir: rustc_data_structures::steal::Steal<rustc_middle::mir::Body<'tcx>>,
            [decode] mir: rustc_middle::mir::Body<'tcx>,
            [] steal_promoted:
                rustc_data_structures::steal::Steal<
                    rustc_index::vec::IndexVec<
                        rustc_middle::mir::Promoted,
                        rustc_middle::mir::Body<'tcx>
                    >
                >,
            [decode] promoted:
                rustc_index::vec::IndexVec<
                    rustc_middle::mir::Promoted,
                    rustc_middle::mir::Body<'tcx>
                >,
            [decode] typeck_results: rustc_middle::ty::TypeckResults<'tcx>,
            [decode] borrowck_result:
                rustc_middle::mir::BorrowCheckResult<'tcx>,
            [decode] unsafety_check_result: rustc_middle::mir::UnsafetyCheckResult,
            [decode] code_region: rustc_middle::mir::coverage::CodeRegion,
            [] const_allocs: rustc_middle::mir::interpret::Allocation,
            // Required for the incremental on-disk cache
            [] mir_keys: rustc_hir::def_id::DefIdSet,
            [] region_scope_tree: rustc_middle::middle::region::ScopeTree,
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
            [] all_traits: Vec<rustc_hir::def_id::DefId>,
            [] privacy_access_levels: rustc_middle::middle::privacy::AccessLevels,
            [] foreign_module: rustc_session::cstore::ForeignModule,
            [] foreign_modules: Vec<rustc_session::cstore::ForeignModule>,
            [] upvars_mentioned: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>,
            [] object_safety_violations: rustc_middle::traits::ObjectSafetyViolation,
            [] codegen_unit: rustc_middle::mir::mono::CodegenUnit<'tcx>,
            [] attribute: rustc_ast::Attribute,
            [] name_set: rustc_data_structures::fx::FxHashSet<rustc_span::symbol::Symbol>,
            [] hir_id_set: rustc_hir::HirIdSet,

            // Interned types
            [] tys: rustc_middle::ty::TyS<'tcx>,
            [] predicates: rustc_middle::ty::PredicateInner<'tcx>,

            // Note that this deliberately duplicates items in the `rustc_hir::arena`,
            // since we need to allocate this type on both the `rustc_hir` arena
            // (during lowering) and the `librustc_middle` arena (for decoding MIR)
            [decode] asm_template: rustc_ast::InlineAsmTemplatePiece,

            // This is used to decode the &'tcx [Span] for InlineAsm's line_spans.
            [decode] span: rustc_span::Span,
            [decode] used_trait_imports: rustc_data_structures::fx::FxHashSet<rustc_hir::def_id::LocalDefId>,

            [] dep_kind: rustc_middle::dep_graph::DepKindStruct,
        ]);
    )
}

arena_types!(rustc_arena::declare_arena);
