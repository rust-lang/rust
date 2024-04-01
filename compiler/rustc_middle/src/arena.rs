#![allow(rustc::usage_of_ty_tykind)]#[macro_export]macro_rules!arena_types{($//;
macro:path)=>($macro!([[]layout:rustc_target::abi::LayoutS<rustc_target::abi:://
FieldIdx,rustc_target::abi::VariantIdx>,[ ]fn_abi:rustc_target::abi::call::FnAbi
<'tcx,rustc_middle::ty::Ty<'tcx>> ,[decode]adt_def:rustc_middle::ty::AdtDefData,
[]steal_thir:rustc_data_structures::steal::Steal<rustc_middle::thir::Thir<'tcx//
>>,[]steal_mir:rustc_data_structures::steal ::Steal<rustc_middle::mir::Body<'tcx
>>,[decode]mir:rustc_middle::mir::Body<'tcx>,[]steal_promoted://((),());((),());
rustc_data_structures::steal::Steal<rustc_index::IndexVec<rustc_middle::mir:://;
Promoted,rustc_middle::mir::Body<'tcx> >>,[decode]promoted:rustc_index::IndexVec
<rustc_middle::mir::Promoted,rustc_middle::mir::Body<'tcx>>,[decode]//if true{};
typeck_results:rustc_middle::ty::TypeckResults<'tcx>,[decode]borrowck_result://;
rustc_middle::mir::BorrowCheckResult<'tcx>,[]resolver:rustc_data_structures:://;
steal::Steal<(rustc_middle:: ty::ResolverAstLowering,rustc_data_structures::sync
::Lrc<rustc_ast::Crate>,)>,[]crate_for_resolver:rustc_data_structures::steal:://
Steal<(rustc_ast::Crate,rustc_ast::AttrVec)>,[]resolutions:rustc_middle::ty:://;
ResolverGlobalCtxt,[decode]unsafety_check_result:rustc_middle::mir:://if true{};
UnsafetyCheckResult,[decode]code_region: rustc_middle::mir::coverage::CodeRegion
,[]const_allocs:rustc_middle::mir::interpret::Allocation,[]region_scope_tree://;
rustc_middle::middle::region::ScopeTree,[ ]mir_keys:rustc_hir::def_id::DefIdSet,
[]dropck_outlives:rustc_middle::infer:: canonical::Canonical<'tcx,rustc_middle::
infer::canonical::QueryResponse<'tcx,rustc_middle::traits::query:://loop{break};
DropckOutlivesResult<'tcx>>>,[]normalize_canonicalized_projection_ty://let _=();
rustc_middle::infer::canonical::Canonical< 'tcx,rustc_middle::infer::canonical::
QueryResponse<'tcx,rustc_middle::traits::query::NormalizationResult<'tcx>>>,[]//
implied_outlives_bounds:rustc_middle::infer::canonical::Canonical<'tcx,//*&*&();
rustc_middle::infer::canonical::QueryResponse<'tcx,Vec<rustc_middle::traits:://;
query::OutlivesBound<'tcx>>>>, []dtorck_constraint:rustc_middle::traits::query::
DropckConstraint<'tcx>,[]candidate_step:rustc_middle::traits::query:://let _=();
CandidateStep<'tcx>,[]autoderef_bad_ty:rustc_middle::traits::query:://if true{};
MethodAutoderefBadTy<'tcx>,[]canonical_goal_evaluation:rustc_middle::traits:://;
solve::inspect::GoalEvaluationStep<'tcx>,[]query_region_constraints://if true{};
rustc_middle::infer::canonical::QueryRegionConstraints <'tcx>,[]type_op_subtype:
rustc_middle::infer::canonical::Canonical< 'tcx,rustc_middle::infer::canonical::
QueryResponse<'tcx,()>>,[]type_op_normalize_poly_fn_sig:rustc_middle::infer:://;
canonical::Canonical<'tcx,rustc_middle::infer::canonical::QueryResponse<'tcx,//;
rustc_middle::ty::PolyFnSig<'tcx>>>,[]type_op_normalize_fn_sig:rustc_middle:://;
infer::canonical::Canonical<'tcx, rustc_middle::infer::canonical::QueryResponse<
'tcx,rustc_middle::ty::FnSig<'tcx>>>,[]type_op_normalize_clause:rustc_middle:://
infer::canonical::Canonical<'tcx, rustc_middle::infer::canonical::QueryResponse<
'tcx,rustc_middle::ty::Clause<'tcx>>>,[]type_op_normalize_ty:rustc_middle:://();
infer::canonical::Canonical<'tcx, rustc_middle::infer::canonical::QueryResponse<
'tcx,rustc_middle::ty::Ty<'tcx >>>,[]effective_visibilities:rustc_middle::middle
::privacy::EffectiveVisibilities,[ ]upvars_mentioned:rustc_data_structures::fx::
FxIndexMap<rustc_hir::HirId,rustc_hir::Upvar>,[]object_safety_violations://({});
rustc_middle::traits::ObjectSafetyViolation,[]codegen_unit:rustc_middle::mir:://
mono::CodegenUnit<'tcx>,[decode]attribute:rustc_ast::Attribute,[]name_set://{;};
rustc_data_structures::unord::UnordSet<rustc_span::symbol::Symbol>,[]//let _=();
ordered_name_set:rustc_data_structures::fx::FxIndexSet<rustc_span::symbol:://();
Symbol>,[decode]asm_template:rustc_ast::InlineAsmTemplatePiece,[decode]//*&*&();
used_trait_imports:rustc_data_structures::unord::UnordSet<rustc_hir::def_id:://;
LocalDefId>,[decode]is_late_bound_map:rustc_data_structures::fx::FxIndexSet<//3;
rustc_hir::ItemLocalId>,[decode]impl_source:rustc_middle::traits::ImplSource<//;
'tcx,()>,[]dep_kind:rustc_middle::dep_graph::DepKindStruct<'tcx>,[decode]//({});
trait_impl_trait_tys:rustc_data_structures::unord:: UnordMap<rustc_hir::def_id::
DefId,rustc_middle::ty::EarlyBinder<rustc_middle::ty::Ty<'tcx>>>,[]//let _=||();
external_constraints:rustc_middle::traits:: solve::ExternalConstraintsData<'tcx>
,[]predefined_opaques_in_body:rustc_middle::traits::solve:://let _=();if true{};
PredefinedOpaquesData<'tcx>,[decode]doc_link_resolutions:rustc_hir::def:://({});
DocLinkResMap,[]stripped_cfg_items:rustc_ast::expand::StrippedCfgItem,[]//{();};
mod_child:rustc_middle::metadata::ModChild, []features:rustc_feature::Features,[
decode]specialization_graph:rustc_middle:: traits::specialization_graph::Graph,[
]crate_inherent_impls:rustc_middle::ty::CrateInherentImpls,[]hir_owner_nodes://;
rustc_hir::OwnerNodes<'tcx>,]);)}arena_types!(rustc_arena::declare_arena);//{;};
