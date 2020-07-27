/// This declares a list of types which can be allocated by `Arena`.
///
/// The `few` modifier will cause allocation to use the shared arena and recording the destructor.
/// This is faster and more memory efficient if there's only a few allocations of the type.
/// Leaving `few` out will cause the type to get its own dedicated `TypedArena` which is
/// faster and more memory efficient if there is lots of allocations.
///
/// Specifying the `decode` modifier will add decode impls for `&T` and `&[T]` where `T` is the type
/// listed. These impls will appear in the implement_ty_decoder! macro.
#[macro_export]
macro_rules! arena_types {
    ($macro:path, $args:tt, $tcx:lifetime) => (
        $macro!($args, [
            [] layouts: rustc_target::abi::Layout, rustc_target::abi::Layout;
            // AdtDef are interned and compared by address
            [] adt_def: rustc_middle::ty::AdtDef, rustc_middle::ty::AdtDef;
            [] steal_mir:
                rustc_middle::ty::steal::Steal<rustc_middle::mir::Body<$tcx>>,
                rustc_middle::ty::steal::Steal<rustc_middle::mir::Body<$tcx>>;
            [decode] mir: rustc_middle::mir::Body<$tcx>, rustc_middle::mir::Body<'_x>;
            [] steal_promoted:
                rustc_middle::ty::steal::Steal<
                    rustc_index::vec::IndexVec<
                        rustc_middle::mir::Promoted,
                        rustc_middle::mir::Body<$tcx>
                    >
                >,
                rustc_middle::ty::steal::Steal<
                    rustc_index::vec::IndexVec<
                        rustc_middle::mir::Promoted,
                        rustc_middle::mir::Body<$tcx>
                    >
                >;
            [decode] promoted:
                rustc_index::vec::IndexVec<
                    rustc_middle::mir::Promoted,
                    rustc_middle::mir::Body<$tcx>
                >,
                rustc_index::vec::IndexVec<
                    rustc_middle::mir::Promoted,
                    rustc_middle::mir::Body<'_x>
                >;
            [decode] typeck_results: rustc_middle::ty::TypeckResults<$tcx>, rustc_middle::ty::TypeckResults<'_x>;
            [decode] borrowck_result:
                rustc_middle::mir::BorrowCheckResult<$tcx>,
                rustc_middle::mir::BorrowCheckResult<'_x>;
            [decode] unsafety_check_result: rustc_middle::mir::UnsafetyCheckResult, rustc_middle::mir::UnsafetyCheckResult;
            [] const_allocs: rustc_middle::mir::interpret::Allocation, rustc_middle::mir::interpret::Allocation;
            // Required for the incremental on-disk cache
            [few, decode] mir_keys: rustc_hir::def_id::DefIdSet, rustc_hir::def_id::DefIdSet;
            [] region_scope_tree: rustc_middle::middle::region::ScopeTree, rustc_middle::middle::region::ScopeTree;
            [] dropck_outlives:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        rustc_middle::traits::query::DropckOutlivesResult<'tcx>
                    >
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y,
                        rustc_middle::traits::query::DropckOutlivesResult<'_z>
                    >
                >;
            [] normalize_projection_ty:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        rustc_middle::traits::query::NormalizationResult<'tcx>
                    >
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y,
                        rustc_middle::traits::query::NormalizationResult<'_z>
                    >
                >;
            [] implied_outlives_bounds:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx,
                        Vec<rustc_middle::traits::query::OutlivesBound<'tcx>>
                    >
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y,
                        Vec<rustc_middle::traits::query::OutlivesBound<'_z>>
                    >
                >;
            [] type_op_subtype:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, ()>
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y, ()>
                >;
            [] type_op_normalize_poly_fn_sig:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::PolyFnSig<'tcx>>
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y, rustc_middle::ty::PolyFnSig<'_z>>
                >;
            [] type_op_normalize_fn_sig:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::FnSig<'tcx>>
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y, rustc_middle::ty::FnSig<'_z>>
                >;
            [] type_op_normalize_predicate:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Predicate<'tcx>>
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y, rustc_middle::ty::Predicate<'_z>>
                >;
            [] type_op_normalize_ty:
                rustc_middle::infer::canonical::Canonical<'tcx,
                    rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Ty<'tcx>>
                >,
                rustc_middle::infer::canonical::Canonical<'_x,
                    rustc_middle::infer::canonical::QueryResponse<'_y, &'_z rustc_middle::ty::TyS<'_w>>
                >;
            [few] all_traits: Vec<rustc_hir::def_id::DefId>, Vec<rustc_hir::def_id::DefId>;
            [few] privacy_access_levels: rustc_middle::middle::privacy::AccessLevels, rustc_middle::middle::privacy::AccessLevels;
            [few] foreign_module: rustc_middle::middle::cstore::ForeignModule, rustc_middle::middle::cstore::ForeignModule;
            [few] foreign_modules: Vec<rustc_middle::middle::cstore::ForeignModule>, Vec<rustc_middle::middle::cstore::ForeignModule>;
            [] upvars_mentioned: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>, rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>;
            [] object_safety_violations: rustc_middle::traits::ObjectSafetyViolation, rustc_middle::traits::ObjectSafetyViolation;
            [] codegen_unit: rustc_middle::mir::mono::CodegenUnit<$tcx>, rustc_middle::mir::mono::CodegenUnit<'_x>;
            [] attribute: rustc_ast::ast::Attribute, rustc_ast::ast::Attribute;
            [] name_set: rustc_data_structures::fx::FxHashSet<rustc_span::symbol::Symbol>, rustc_data_structures::fx::FxHashSet<rustc_span::symbol::Symbol>;
            [] hir_id_set: rustc_hir::HirIdSet, rustc_hir::HirIdSet;

            // Interned types
            [] tys: rustc_middle::ty::TyS<$tcx>, rustc_middle::ty::TyS<'_x>;
            [] predicates: rustc_middle::ty::PredicateInner<$tcx>, rustc_middle::ty::PredicateInner<'_x>;

            // HIR query types
            [few] indexed_hir: rustc_middle::hir::map::IndexedHir<$tcx>, rustc_middle::hir::map::IndexedHir<'_x>;
            [few] hir_definitions: rustc_hir::definitions::Definitions, rustc_hir::definitions::Definitions;
            [] hir_owner: rustc_middle::hir::Owner<$tcx>, rustc_middle::hir::Owner<'_x>;
            [] hir_owner_nodes: rustc_middle::hir::OwnerNodes<$tcx>, rustc_middle::hir::OwnerNodes<'_x>;

            // Note that this deliberately duplicates items in the `rustc_hir::arena`,
            // since we need to allocate this type on both the `rustc_hir` arena
            // (during lowering) and the `librustc_middle` arena (for decoding MIR)
            [decode] asm_template: rustc_ast::ast::InlineAsmTemplatePiece, rustc_ast::ast::InlineAsmTemplatePiece;

            // This is used to decode the &'tcx [Span] for InlineAsm's line_spans.
            [decode] span: rustc_span::Span, rustc_span::Span;
            [decode] used_trait_imports: rustc_data_structures::fx::FxHashSet<rustc_hir::def_id::LocalDefId>, rustc_data_structures::fx::FxHashSet<rustc_hir::def_id::LocalDefId>;
        ], $tcx);
    )
}

arena_types!(rustc_arena::declare_arena, [], 'tcx);
