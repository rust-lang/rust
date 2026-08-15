//! Declares [`rustc_middle::arena::Arena`], which can allocate values of any
//! `Copy` type, and any `!Copy` type explicitly listed below.

use rustc_serialize::Decodable;

use crate::ty::codec::{RefDecodable, TyDecoder};
use crate::ty::{Ty, TyCtxt};

// If a type `T` supported by the arena also needs to support decoding into `&'tcx T`
// backed by an arena allocation (via `RefDecodable`), add it to the list in
// `impl_ref_decodable_into_arena!`.

rustc_arena::declare_arena! {
    layout: rustc_abi::LayoutData<rustc_abi::FieldIdx, rustc_abi::VariantIdx>,
    proxy_coroutine_layout: rustc_middle::mir::CoroutineLayout<'tcx>,
    fn_abi: rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
    adt_def: rustc_middle::ty::AdtDefData,
    steal_thir: rustc_data_structures::steal::Steal<rustc_middle::thir::Thir<'tcx>>,
    steal_mir: rustc_data_structures::steal::Steal<rustc_middle::mir::Body<'tcx>>,
    mir: rustc_middle::mir::Body<'tcx>,
    steal_promoted:
        rustc_data_structures::steal::Steal<
            rustc_index::IndexVec<
                rustc_middle::mir::Promoted,
                rustc_middle::mir::Body<'tcx>
            >
        >,
    promoted:
        rustc_index::IndexVec<
            rustc_middle::mir::Promoted,
            rustc_middle::mir::Body<'tcx>
        >,
    typeck_results: rustc_middle::ty::TypeckResults<'tcx>,
    borrowck_result: rustc_middle::mir::BorrowCheckResult<'tcx>,
    resolver: rustc_data_structures::steal::Steal<rustc_middle::ty::ResolverAstLowering<'tcx>>,
    index_ast:
        rustc_index::IndexVec<
            rustc_span::def_id::LocalDefId,
            rustc_data_structures::steal::Steal<(
                std::sync::Arc<rustc_middle::ty::ResolverAstLowering<'tcx>>,
                rustc_ast::AstOwner
            )>
        >,
    crate_alone: rustc_data_structures::steal::Steal<rustc_ast::Crate>,
    crate_for_resolver: rustc_data_structures::steal::Steal<(rustc_ast::Crate, rustc_ast::AttrVec)>,
    resolutions: rustc_middle::ty::ResolverGlobalCtxt,
    const_allocs: rustc_middle::mir::interpret::Allocation,
    region_scope_tree: rustc_middle::middle::region::ScopeTree,
    // Required for the incremental on-disk cache
    mir_keys: rustc_hir::def_id::DefIdSet,
    dropck_outlives:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx,
                rustc_middle::traits::query::DropckOutlivesResult<'tcx>
            >
        >,
    normalize_canonicalized_projection:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx,
                rustc_middle::traits::query::NormalizationResult<'tcx>
            >
        >,
    implied_outlives_bounds:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx,
                Vec<rustc_middle::traits::query::OutlivesBound<'tcx>>
            >
        >,
    dtorck_constraint: rustc_middle::traits::query::DropckConstraint<'tcx>,
    candidate_step: rustc_middle::traits::query::CandidateStep<'tcx>,
    autoderef_bad_ty: rustc_middle::traits::query::MethodAutoderefBadTy<'tcx>,
    query_region_constraints: rustc_middle::infer::canonical::QueryRegionConstraints<'tcx>,
    type_op_subtype:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx, ()>
        >,
    type_op_normalize_poly_fn_sig:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::PolyFnSig<'tcx>>
        >,
    type_op_normalize_fn_sig:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::FnSig<'tcx>>
        >,
    type_op_normalize_clause:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx, rustc_middle::ty::Clause<'tcx>>
        >,
    type_op_normalize_ty:
        rustc_middle::infer::canonical::Canonical<'tcx,
            rustc_middle::infer::canonical::QueryResponse<'tcx, Ty<'tcx>>
        >,
    inspect_probe: rustc_middle::traits::solve::inspect::Probe<TyCtxt<'tcx>>,
    effective_visibilities: rustc_middle::middle::privacy::EffectiveVisibilities,
    upvars_mentioned: rustc_data_structures::fx::FxIndexMap<rustc_hir::HirId, rustc_hir::Upvar>,
    dyn_compatibility_violations: rustc_middle::traits::DynCompatibilityViolation,
    codegen_unit: rustc_middle::mono::CodegenUnit<'tcx>,
    attribute: rustc_hir::Attribute,
    name_set: rustc_data_structures::unord::UnordSet<rustc_span::Symbol>,
    autodiff_item: rustc_hir::attrs::AutoDiffItem,
    ordered_name_set: rustc_data_structures::fx::FxIndexSet<rustc_span::Symbol>,
    stable_order_of_exportable_impls:
        rustc_data_structures::fx::FxIndexMap<rustc_hir::def_id::DefId, usize>,

    // Note that this deliberately duplicates items in the `rustc_hir::arena`,
    // since we need to allocate this type on both the `rustc_hir` arena
    // (during lowering) and the `rustc_middle` arena (for decoding MIR)
    asm_template: rustc_ast::InlineAsmTemplatePiece,
    used_trait_imports: rustc_data_structures::unord::UnordSet<rustc_hir::def_id::LocalDefId>,
    is_late_bound_map: rustc_data_structures::fx::FxIndexSet<rustc_hir::ItemLocalId>,
    impl_source: rustc_middle::traits::ImplSource<'tcx, ()>,

    dep_kind_vtable: rustc_middle::dep_graph::DepKindVTable<'tcx>,

    trait_impl_trait_tys:
        rustc_data_structures::unord::UnordMap<
            rustc_hir::def_id::DefId,
            rustc_middle::ty::EarlyBinder<'tcx, Ty<'tcx>>
        >,
    external_constraints: rustc_middle::traits::solve::ExternalConstraintsData<TyCtxt<'tcx>>,
    doc_link_resolutions: rustc_hir::def::DocLinkResMap,
    stripped_cfg_items: rustc_hir::attrs::StrippedCfgItem,
    mod_child: rustc_middle::metadata::ModChild,
    features: rustc_feature::Features,
    specialization_graph: rustc_middle::traits::specialization_graph::Graph,
    crate_inherent_impls: rustc_middle::ty::CrateInherentImpls,
    hir_owner_nodes: rustc_hir::OwnerNodes<'tcx>,
    token_stream: rustc_ast::tokenstream::TokenStream,
    maybe_owner: rustc_middle::hir::ProjectedMaybeOwner<'tcx>,
    owner_info: rustc_middle::hir::ProjectedOwnerInfo<'tcx>,
    parenting: rustc_hir::def_id::LocalDefIdMap<rustc_hir::ItemLocalId>,
    trait_candidates: rustc_hir::ItemLocalMap<&'tcx [rustc_hir::TraitCandidate<'tcx>]>,
    delayed_lints: rustc_data_structures::steal::Steal<rustc_hir::lints::DelayedLints>,
}

#[inline]
fn decode_arena_allocatable<'tcx, D, C, T>(decoder: &mut D) -> &'tcx T
where
    D: TyDecoder<'tcx>,
    T: ArenaAllocatable<'tcx, C> + Decodable<D>,
{
    let value: T = Decodable::decode(decoder);
    decoder.interner().arena.alloc(value)
}

#[inline]
fn decode_arena_allocatable_slice<'tcx, D, C, T>(decoder: &mut D) -> &'tcx [T]
where
    D: TyDecoder<'tcx>,
    T: ArenaAllocatable<'tcx, C> + Decodable<D>,
{
    let values: Vec<T> = Decodable::decode(decoder);
    decoder.interner().arena.alloc_from_iter(values)
}

macro_rules! impl_ref_decodable_into_arena {
    (
        $(
            $ty:ty,
        )*
    ) => {
        $(
            impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for $ty {
                #[inline]
                fn decode(decoder: &mut D) -> &'tcx Self {
                    decode_arena_allocatable(decoder)
                }
            }

            impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for [$ty] {
                #[inline]
                fn decode(decoder: &mut D) -> &'tcx Self {
                    decode_arena_allocatable_slice(decoder)
                }
            }
        )*
    }
}

// For each of these types, implements `RefDecodable` for `T` (and `[T]`) by
// decoding to `T` and then moving the value or values into an arena allocation.
//
// Types in this list must be `ArenaAllocatable`, either because they are `Copy`
// or because they are listed in the `declare_arena!` invocation.
impl_ref_decodable_into_arena! {
    // tidy-alphabetical-start
    (rustc_middle::middle::exported_symbols::ExportedSymbol<'tcx>, rustc_middle::middle::exported_symbols::SymbolExportInfo),
    rustc_ast::InlineAsmTemplatePiece,
    rustc_ast::tokenstream::TokenStream,
    rustc_data_structures::unord::UnordMap<rustc_span::def_id::DefId, rustc_middle::ty::EarlyBinder<'tcx, Ty<'tcx>>>,
    rustc_data_structures::unord::UnordSet<rustc_span::def_id::LocalDefId>,
    rustc_hir::Attribute,
    rustc_index::IndexVec<rustc_middle::mir::Promoted, rustc_middle::mir::Body<'tcx>>,
    rustc_middle::middle::deduced_param_attrs::DeducedParamAttrs,
    rustc_middle::mir::Body<'tcx>,
    rustc_middle::traits::ImplSource<'tcx, ()>,
    rustc_middle::traits::specialization_graph::Graph,
    rustc_middle::ty::TypeckResults<'tcx>,
    rustc_middle::ty::Variance,
    rustc_span::Ident,
    rustc_span::Span,
    rustc_span::def_id::DefId,
    rustc_span::def_id::LocalDefId,
    // tidy-alphabetical-end
}
