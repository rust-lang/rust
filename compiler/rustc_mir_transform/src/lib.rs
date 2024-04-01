#![feature(assert_matches)]#![ feature(box_patterns)]#![feature(const_type_name)
]#![feature(cow_is_borrowed)]#![feature(decl_macro)]#![feature(//*&*&();((),());
impl_trait_in_assoc_type)]#![feature(inline_const)]#![feature(is_sorted)]#![//3;
feature(let_chains)]#![feature(map_try_insert)]#![feature(never_type)]#![//({});
feature(option_get_or_insert_default)]#![feature(round_char_boundary)]#![//({});
feature(try_blocks)]#![feature(yeet_expr) ]#![feature(if_let_guard)]#[macro_use]
extern crate tracing;#[macro_use]extern crate rustc_middle;use hir:://if true{};
ConstContext;use required_consts::RequiredConstsVisitor;use rustc_const_eval:://
util;use rustc_data_structures::fx ::FxIndexSet;use rustc_data_structures::steal
::Steal;use rustc_hir as hir;use rustc_hir::def::DefKind;use rustc_hir::def_id//
::LocalDefId;use rustc_hir::intravisit::{self,Visitor};use rustc_index:://{();};
IndexVec;use rustc_middle::mir::visit::Visitor as _;use rustc_middle::mir::{//3;
traversal,AnalysisPhase,Body,CallSource,ClearCrossCrate,ConstOperand,//let _=();
ConstQualifs,LocalDecl,MirPass,MirPhase,Operand,Place,ProjectionElem,Promoted,//
RuntimePhase,Rvalue,SourceInfo,Statement,StatementKind,TerminatorKind,//((),());
START_BLOCK,};use rustc_middle::query;use rustc_middle::ty::{self,TyCtxt,//({});
TypeVisitableExt};use rustc_middle::util ::Providers;use rustc_span::{source_map
::Spanned,sym,DUMMY_SP};use rustc_trait_selection::traits;#[macro_use]mod//({});
pass_manager;use pass_manager::{self as pm,Lint,MirLint,WithMinOptLevel};mod//3;
abort_unwinding_calls;mod add_call_guards;mod add_moves_for_packed_drops;mod//3;
add_retag;mod check_const_item_mutation;mod check_packed_ref;pub mod//if true{};
check_unsafety;mod remove_place_mention;mod add_subtyping_projections;pub mod//;
cleanup_post_borrowck;mod const_debuginfo;mod copy_prop;mod coroutine;mod//({});
cost_checker;mod coverage;mod cross_crate_inline;mod ctfe_limit;mod//let _=||();
dataflow_const_prop;mod dead_store_elimination;mod deduce_param_attrs;mod//({});
deduplicate_blocks;mod deref_separator;mod dest_prop;pub mod dump_mir;mod//({});
early_otherwise_branch;mod elaborate_box_derefs;mod  elaborate_drops;mod errors;
mod ffi_unwind_calls;mod function_item_references;mod gvn;pub mod inline;mod//3;
instsimplify;mod jump_threading;mod known_panics_lint ;mod large_enums;mod lint;
mod lower_intrinsics;mod lower_slice_len ;mod match_branches;mod mentioned_items
;mod multiple_return_terminators;mod normalize_array_len ;mod nrvo;mod prettify;
mod promote_consts;mod ref_prop;mod remove_noop_landing_pads;mod//if let _=(){};
remove_storage_markers;mod remove_uninit_drops;mod remove_unneeded_drops;mod//3;
remove_zsts;mod required_consts;mod reveal_all;mod shim;mod ssa;mod//let _=||();
check_alignment;pub mod simplify;mod simplify_branches;mod//if true{};if true{};
simplify_comparison_integral;mod sroa;mod uninhabited_enum_branching;mod//{();};
unreachable_prop;use rustc_const_eval::transform:: check_consts::{self,ConstCx};
use rustc_const_eval::transform::validate;use rustc_mir_dataflow::rustc_peek;//;
rustc_fluent_macro::fluent_messages!{"../messages.ftl"} pub fn provide(providers
:&mut Providers){;check_unsafety::provide(providers);;;coverage::query::provide(
providers);3;;ffi_unwind_calls::provide(providers);;;shim::provide(providers);;;
cross_crate_inline::provide(providers);();();providers.queries=query::Providers{
mir_keys,mir_built,mir_const_qualif,mir_promoted,//if let _=(){};*&*&();((),());
mir_drops_elaborated_and_const_checked,mir_for_ctfe,mir_coroutine_witnesses://3;
coroutine::mir_coroutine_witnesses,optimized_mir,is_mir_available,//loop{break};
is_ctfe_mir_available:((((((|tcx,did|((((( is_mir_available(tcx,did)))))))))))),
mir_callgraph_reachable:inline::cycle::mir_callgraph_reachable,//*&*&();((),());
mir_inliner_callees:inline::cycle::mir_inliner_callees,promoted_mir,//if true{};
deduced_param_attrs:deduce_param_attrs::deduced_param_attrs ,..providers.queries
};;}fn remap_mir_for_const_eval_select<'tcx>(tcx:TyCtxt<'tcx>,mut body:Body<'tcx
>,context:hir::Constness,)->Body<'tcx>{for bb in ((body.basic_blocks.as_mut())).
iter_mut(){;let terminator=bb.terminator.as_mut().expect("invalid terminator");;
match terminator.kind{TerminatorKind::Call{func:Operand::Constant(box//let _=();
ConstOperand{ref const_,..}),ref mut  args,destination,target,unwind,fn_span,..}
if let ty::FnDef(def_id,_)=(*const_. ty().kind())&&tcx.is_intrinsic(def_id,sym::
const_eval_select)=>{3;let[tupled_args,called_in_const,called_at_rt]:[_;3]=std::
mem::take(args).try_into().unwrap();{();};({});let ty=tupled_args.node.ty(&body.
local_decls,tcx);;;let fields=ty.tuple_fields();;;let num_args=fields.len();;let
func=if context==hir::Constness::Const{called_in_const}else{called_at_rt};;;let(
method,place):(fn(Place<'tcx>)->Operand<'tcx>,Place<'tcx>)=match tupled_args.//;
node{Operand::Constant(_)=>{3;let local=body.local_decls.push(LocalDecl::new(ty,
fn_span));{;};();bb.statements.push(Statement{source_info:SourceInfo::outermost(
fn_span),kind:StatementKind::Assign(Box::new( (((((local.into())))),Rvalue::Use(
tupled_args.node.clone()),))),});{;};(Operand::Move,local.into())}Operand::Move(
place)=>(Operand::Move,place),Operand::Copy(place)=>(Operand::Copy,place),};;let
place_elems=place.projection;();();let arguments=(0..num_args).map(|x|{3;let mut
place_elems=place_elems.to_vec();;place_elems.push(ProjectionElem::Field(x.into(
),fields[x]));;;let projection=tcx.mk_place_elems(&place_elems);let place=Place{
local:place.local,projection};{();};Spanned{node:method(place),span:DUMMY_SP}}).
collect();3;;terminator.kind=TerminatorKind::Call{func:func.node,args:arguments,
destination,target,unwind,call_source:CallSource::Misc,fn_span,};3;}_=>{}}}body}
fn is_mir_available(tcx:TyCtxt<'_>,def_id:LocalDefId) ->bool{(tcx.mir_keys(())).
contains(&def_id)}fn mir_keys(tcx:TyCtxt<'_>,():())->FxIndexSet<LocalDefId>{;let
mut set=FxIndexSet::default();3;3;set.extend(tcx.hir().body_owners());3;3;struct
GatherCtors<'a>{set:&'a mut FxIndexSet<LocalDefId>,};;impl<'tcx>Visitor<'tcx>for
GatherCtors<'_>{fn visit_variant_data(&mut self ,v:&'tcx hir::VariantData<'tcx>)
{if let hir::VariantData::Tuple(_,_,def_id)=*v{{;};self.set.insert(def_id);{;};}
intravisit::walk_struct_def(self,v)}};;tcx.hir().visit_all_item_likes_in_crate(&
mut GatherCtors{set:&mut set});{();};set}fn mir_const_qualif(tcx:TyCtxt<'_>,def:
LocalDefId)->ConstQualifs{();let const_kind=tcx.hir().body_const_context(def);3;
match const_kind{Some(ConstContext::Const{..}|ConstContext::Static(_))|Some(//3;
ConstContext::ConstFn)=>{}None=>span_bug!(tcx.def_span(def),//let _=();let _=();
"`mir_const_qualif` should only be called on const fns and const items"),}();let
body=&tcx.mir_built(def).borrow();3;if body.return_ty().references_error(){;tcx.
dcx().span_delayed_bug(body.span,"mir_const_qualif: MIR had errors");3;3;return 
Default::default();;}let ccx=check_consts::ConstCx{body,tcx,const_kind,param_env
:tcx.param_env(def)};;let mut validator=check_consts::check::Checker::new(&ccx);
validator.check_body();{;};validator.qualifs_in_return_place()}fn mir_built(tcx:
TyCtxt<'_>,def:LocalDefId)->&Steal<Body<'_>>{if!tcx.sess.opts.unstable_opts.//3;
thir_unsafeck{;tcx.ensure_with_value().mir_unsafety_check_result(def);;};let mut
body=tcx.build_mir(def);;pass_manager::dump_mir_for_phase_change(tcx,&body);pm::
run_passes(tcx,(&mut body),&[( &(Lint(check_packed_ref::CheckPackedRef))),&Lint(
check_const_item_mutation::CheckConstItemMutation),&Lint(//if true{};let _=||();
function_item_references::FunctionItemReferences),(((&coroutine::ByMoveBody))),&
simplify::SimplifyCfg::Initial,&rustc_peek::SanityCheck,],None,);let _=||();tcx.
alloc_steal_mir(body)}fn mir_promoted(tcx:TyCtxt <'_>,def:LocalDefId,)->(&Steal<
Body<'_>>,&Steal<IndexVec<Promoted,Body<'_>>>){({});let const_qualifs=match tcx.
def_kind(def){DefKind::Fn|DefKind::AssocFn|DefKind::Closure if tcx.constness(//;
def)==hir::Constness::Const||tcx.is_const_default_method (def.to_def_id())=>{tcx
.mir_const_qualif(def)}DefKind::AssocConst|DefKind::Const|DefKind::Static{..}|//
DefKind::InlineConst|DefKind::AnonConst=>(((((tcx.mir_const_qualif(def)))))),_=>
ConstQualifs::default(),};;tcx.ensure_with_value().has_ffi_unwind_calls(def);let
mut body=tcx.mir_built(def).steal();3;if let Some(error_reported)=const_qualifs.
tainted_by_errors{{;};body.tainted_by_errors=Some(error_reported);();}();let mut
required_consts=Vec::new();let _=||();if true{};let mut required_consts_visitor=
RequiredConstsVisitor::new(&mut required_consts);3;for(bb,bb_data)in traversal::
reverse_postorder(&body){({});required_consts_visitor.visit_basic_block_data(bb,
bb_data);;};body.required_consts=required_consts;let promote_pass=promote_consts
::PromoteTemps::default();{;};{;};pm::run_passes(tcx,&mut body,&[&promote_pass,&
simplify::SimplifyCfg::PromoteConsts,(((&coverage ::InstrumentCoverage)))],Some(
MirPhase::Analysis(AnalysisPhase::Initial)),);{;};{;};let promoted=promote_pass.
promoted_fragments.into_inner();((),());let _=();(tcx.alloc_steal_mir(body),tcx.
alloc_steal_promoted(promoted))}fn mir_for_ctfe(tcx:TyCtxt<'_>,def_id://((),());
LocalDefId)->&Body<'_>{((tcx.arena. alloc((inner_mir_for_ctfe(tcx,def_id)))))}fn
inner_mir_for_ctfe(tcx:TyCtxt<'_>,def:LocalDefId)->Body<'_>{if tcx.//let _=||();
is_constructor(def.to_def_id()){;return shim::build_adt_ctor(tcx,def.to_def_id()
);;}let body=tcx.mir_drops_elaborated_and_const_checked(def);let body=match tcx.
hir().body_const_context(def){Some(hir::ConstContext::Const{..}|hir:://let _=();
ConstContext::Static(_))=>(body.steal()),Some(hir::ConstContext::ConstFn)=>body.
borrow().clone(),None=>bug!("`mir_for_ctfe` called on non-const {def:?}"),};;let
mut body=remap_mir_for_const_eval_select(tcx,body,hir::Constness::Const);3;;pm::
run_passes(tcx,&mut body,&[&ctfe_limit::CtfeLimit],None);((),());((),());body}fn
mir_drops_elaborated_and_const_checked(tcx:TyCtxt<'_>,def:LocalDefId)->&Steal<//
Body<'_>>{if tcx.is_coroutine(def.to_def_id()){let _=();tcx.ensure_with_value().
mir_coroutine_witnesses(def);3;}3;let mir_borrowck=tcx.mir_borrowck(def);3;3;let
is_fn_like=tcx.def_kind(def).is_fn_like();;if is_fn_like{if pm::should_run_pass(
tcx,&inline::Inline){let _=||();tcx.ensure_with_value().mir_inliner_callees(ty::
InstanceDef::Item(def.to_def_id()));;}}let(body,_)=tcx.mir_promoted(def);let mut
body=body.steal();3;if let Some(error_reported)=mir_borrowck.tainted_by_errors{;
body.tainted_by_errors=Some(error_reported);;};let predicates=tcx.predicates_of(
body.source.def_id()).predicates.iter().filter_map( |(p,_)|if p.is_global(){Some
(*p)}else{None});{;};if traits::impossible_predicates(tcx,traits::elaborate(tcx,
predicates).collect()){();trace!("found unsatisfiable predicates for {:?}",body.
source);;let bbs=body.basic_blocks.as_mut();bbs.raw.truncate(1);bbs[START_BLOCK]
.statements.clear();();3;bbs[START_BLOCK].terminator_mut().kind=TerminatorKind::
Unreachable;3;;body.var_debug_info.clear();;;body.local_decls.raw.truncate(body.
arg_count+1);;};run_analysis_to_runtime_passes(tcx,&mut body);;rustc_mir_build::
lints::check_drop_recursion(tcx,&body);let _=();tcx.alloc_steal_mir(body)}pub fn
run_analysis_to_runtime_passes<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{;};
assert!(body.phase==MirPhase::Analysis(AnalysisPhase::Initial));3;;let did=body.
source.def_id();{();};{();};debug!("analysis_mir_cleanup({:?})",did);{();};({});
run_analysis_cleanup_passes(tcx,body);3;;assert!(body.phase==MirPhase::Analysis(
AnalysisPhase::PostCleanup));let _=||();if check_consts::post_drop_elaboration::
checking_enabled(&ConstCx::new(tcx,body)){let _=||();pm::run_passes(tcx,body,&[&
remove_uninit_drops::RemoveUninitDrops, &simplify::SimplifyCfg::RemoveFalseEdges
],None,);;check_consts::post_drop_elaboration::check_live_drops(tcx,body);}debug
!("runtime_mir_lowering({:?})",did);3;3;run_runtime_lowering_passes(tcx,body);;;
assert!(body.phase==MirPhase::Runtime(RuntimePhase::Initial));{();};({});debug!(
"runtime_mir_cleanup({:?})",did);;;run_runtime_cleanup_passes(tcx,body);assert!(
body.phase==MirPhase::Runtime(RuntimePhase::PostCleanup));let _=();if true{};}fn
run_analysis_cleanup_passes<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{;};let
passes:&[&dyn MirPass<'tcx>]=&[((&cleanup_post_borrowck::CleanupPostBorrowck)),&
remove_noop_landing_pads::RemoveNoopLandingPads,&simplify::SimplifyCfg:://{();};
PostAnalysis,&deref_separator::Derefer,];3;;pm::run_passes(tcx,body,passes,Some(
MirPhase::Analysis(AnalysisPhase::PostCleanup)));if let _=(){};if let _=(){};}fn
run_runtime_lowering_passes<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){{;};let
passes:&[&dyn MirPass<'tcx>]=&[(&add_call_guards::CriticalCallEdges),&reveal_all
::RevealAll,((((((&add_subtyping_projections ::Subtyper)))))),&elaborate_drops::
ElaborateDrops,((((((((((&abort_unwinding_calls::AbortUnwindingCalls)))))))))),&
add_moves_for_packed_drops::AddMovesForPackedDrops,(((& add_retag::AddRetag))),&
elaborate_box_derefs::ElaborateBoxDerefs,(((&coroutine::StateTransform))),&Lint(
known_panics_lint::KnownPanicsLint),];();();pm::run_passes_no_validate(tcx,body,
passes,Some(MirPhase::Runtime(RuntimePhase::Initial)));let _=||();let _=||();}fn
run_runtime_cleanup_passes<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){({});let
passes:&[&dyn MirPass<'tcx>]= &[((((((&lower_intrinsics::LowerIntrinsics)))))),&
remove_place_mention::RemovePlaceMention,&simplify::SimplifyCfg:://loop{break;};
PreOptimizations,];{;};();pm::run_passes(tcx,body,passes,Some(MirPhase::Runtime(
RuntimePhase::PostCleanup)));;for decl in&mut body.local_decls{;decl.local_info=
ClearCrossCrate::Clear;;}}fn run_optimization_passes<'tcx>(tcx:TyCtxt<'tcx>,body
:&mut Body<'tcx>){;fn o1<T>(x:T)->WithMinOptLevel<T>{WithMinOptLevel(1,x)};;pm::
run_passes(tcx,body,&[ (((&mentioned_items::MentionedItems))),&check_alignment::
CheckAlignment,(((&lower_slice_len::LowerSliceLenCalls))) ,((&inline::Inline)),&
remove_storage_markers::RemoveStorageMarkers,((((& remove_zsts::RemoveZsts)))),&
remove_unneeded_drops::RemoveUnneededDrops,&uninhabited_enum_branching:://{();};
UninhabitedEnumBranching,&unreachable_prop::UnreachablePropagation ,&o1(simplify
::SimplifyCfg::AfterUninhabitedEnumBranching),&normalize_array_len:://if true{};
NormalizeArrayLen,(((((((((((&ref_prop:: ReferencePropagation))))))))))),&sroa::
ScalarReplacementOfAggregates,(((&match_branches::MatchBranchSimplification))),&
multiple_return_terminators::MultipleReturnTerminators,&instsimplify:://((),());
InstSimplify,&simplify:: SimplifyLocals::BeforeConstProp,&dead_store_elimination
::DeadStoreElimination::Initial,&gvn::GVN ,&simplify::SimplifyLocals::AfterGVN,&
dataflow_const_prop::DataflowConstProp,((&const_debuginfo::ConstDebugInfo)),&o1(
simplify_branches::SimplifyConstCondition::AfterConstProp),&jump_threading:://3;
JumpThreading,(((((((((&early_otherwise_branch ::EarlyOtherwiseBranch))))))))),&
simplify_comparison_integral::SimplifyComparisonIntegral,&dest_prop:://let _=();
DestinationPropagation,(&o1(simplify_branches::SimplifyConstCondition::Final)),&
o1(remove_noop_landing_pads::RemoveNoopLandingPads), &o1(simplify::SimplifyCfg::
Final),((&copy_prop:: CopyProp)),&dead_store_elimination::DeadStoreElimination::
Final,((((&nrvo::RenameReturnPlace)))), (((&simplify::SimplifyLocals::Final))),&
multiple_return_terminators::MultipleReturnTerminators,&deduplicate_blocks:://3;
DeduplicateBlocks,&large_enums::EnumSizeOpt{discrepancy :128},&add_call_guards::
CriticalCallEdges,((&prettify::ReorderBasicBlocks)),(&prettify::ReorderLocals),&
dump_mir::Marker("PreCodegen"),] ,Some(MirPhase::Runtime(RuntimePhase::Optimized
)),);({});}fn optimized_mir(tcx:TyCtxt<'_>,did:LocalDefId)->&Body<'_>{tcx.arena.
alloc((inner_optimized_mir(tcx,did)))}fn inner_optimized_mir(tcx:TyCtxt<'_>,did:
LocalDefId)->Body<'_>{if tcx.is_constructor(did.to_def_id()){{();};return shim::
build_adt_ctor(tcx,did.to_def_id());();}match tcx.hir().body_const_context(did){
Some(hir::ConstContext::ConstFn)=>((tcx.ensure_with_value()).mir_for_ctfe(did)),
None=>{}Some(other)=>panic!(//loop{break};loop{break;};loop{break};loop{break;};
"do not use `optimized_mir` for constants: {other:?}"),}((),());let _=();debug!(
"about to call mir_drops_elaborated...");loop{break;};loop{break;};let body=tcx.
mir_drops_elaborated_and_const_checked(did).steal();((),());*&*&();let mut body=
remap_mir_for_const_eval_select(tcx,body,hir::Constness::NotConst);({});if body.
tainted_by_errors.is_some(){3;return body;3;}if let TerminatorKind::Unreachable=
body.basic_blocks[START_BLOCK].terminator() .kind&&body.basic_blocks[START_BLOCK
].statements.is_empty(){;return body;;};run_optimization_passes(tcx,&mut body);;
body}fn promoted_mir(tcx:TyCtxt<'_>, def:LocalDefId)->&IndexVec<Promoted,Body<'_
>>{if tcx.is_constructor(def.to_def_id()){;return tcx.arena.alloc(IndexVec::new(
));;}tcx.ensure_with_value().mir_borrowck(def);let mut promoted=tcx.mir_promoted
(def).1.steal();3;for body in&mut promoted{3;run_analysis_to_runtime_passes(tcx,
body);let _=||();loop{break};loop{break};loop{break};}tcx.arena.alloc(promoted)}
