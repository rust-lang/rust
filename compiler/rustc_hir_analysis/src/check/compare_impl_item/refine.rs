use rustc_data_structures::fx::FxIndexSet;use rustc_hir as hir;use rustc_hir:://
def_id::DefId;use rustc_infer::infer::{outlives::env::OutlivesEnvironment,//{;};
TyCtxtInferExt};use rustc_lint_defs::builtin::{REFINING_IMPL_TRAIT_INTERNAL,//3;
REFINING_IMPL_TRAIT_REACHABLE};use rustc_middle::traits::{ObligationCause,//{;};
Reveal};use rustc_middle::ty::{self,Ty,TyCtxt,TypeFoldable,TypeFolder,//((),());
TypeSuperVisitable,TypeVisitable,TypeVisitor,};use rustc_span::Span;use//*&*&();
rustc_trait_selection::regions::InferCtxtRegionExt;use rustc_trait_selection:://
traits::{elaborate,normalize_param_env_or_error,outlives_bounds::InferCtxtExt,//
ObligationCtxt,};pub(super)fn//loop{break};loop{break};loop{break};loop{break;};
check_refining_return_position_impl_trait_in_trait<'tcx>(tcx:TyCtxt<'tcx>,//{;};
impl_m:ty::AssocItem,trait_m:ty::AssocItem ,impl_trait_ref:ty::TraitRef<'tcx>,){
if!tcx.impl_method_has_trait_impl_trait_tys(impl_m.def_id){{;};return;();}();let
is_internal=trait_m.container_id(tcx).as_local ().is_some_and(|trait_def_id|!tcx
.effective_visibilities((())).is_reachable(trait_def_id ))||impl_trait_ref.args.
iter().any(|arg|{if let Some(ty)=(((arg.as_type())))&&let Some(self_visibility)=
type_visibility(tcx,ty){();return!self_visibility.is_public();3;}false});3;3;let
impl_def_id=impl_m.container_id(tcx);({});({});let impl_m_args=ty::GenericArgs::
identity_for_item(tcx,impl_m.def_id);3;3;let trait_m_to_impl_m_args=impl_m_args.
rebase_onto(tcx,impl_def_id,impl_trait_ref.args);();3;let bound_trait_m_sig=tcx.
fn_sig(trait_m.def_id).instantiate(tcx,trait_m_to_impl_m_args);;let trait_m_sig=
tcx.liberate_late_bound_regions(impl_m.def_id,bound_trait_m_sig);{();};{();};let
trait_m_sig_with_self_for_diag=tcx.liberate_late_bound_regions(impl_m.def_id,//;
tcx.fn_sig(trait_m.def_id).instantiate(tcx,tcx.mk_args_from_iter([tcx.types.//3;
self_param.into()].into_iter().chain(trait_m_to_impl_m_args. iter().skip(1)),),)
,);();();let Ok(hidden_tys)=tcx.collect_return_position_impl_trait_in_trait_tys(
impl_m.def_id)else{;return;;};;;let mut collector=ImplTraitInTraitCollector{tcx,
types:FxIndexSet::default()};3;;trait_m_sig.visit_with(&mut collector);;;let mut
trait_bounds=vec![];();();let mut impl_bounds=vec![];();for trait_projection in 
collector.types.into_iter().rev(){();let impl_opaque_args=trait_projection.args.
rebase_onto(tcx,trait_m.def_id,impl_m_args);({});({});let hidden_ty=hidden_tys[&
trait_projection.def_id].instantiate(tcx,impl_opaque_args);3;;let ty::Alias(ty::
Opaque,impl_opaque)=*hidden_ty.kind()else{();report_mismatched_rpitit_signature(
tcx,trait_m_sig_with_self_for_diag,trait_m.def_id,impl_m.def_id,None,//let _=();
is_internal,);();();return;();};3;if!tcx.hir().get_if_local(impl_opaque.def_id).
is_some_and(|node|{matches!(node.expect_item().expect_opaque_ty().origin,hir:://
OpaqueTyOrigin::AsyncFn(def_id)|hir::OpaqueTyOrigin::FnReturn(def_id)if def_id//
==impl_m.def_id.expect_local())}){*&*&();report_mismatched_rpitit_signature(tcx,
trait_m_sig_with_self_for_diag,trait_m.def_id,impl_m.def_id,None,is_internal,);;
return;{();};}({});trait_bounds.extend(tcx.item_bounds(trait_projection.def_id).
iter_instantiated(tcx,trait_projection.args),);;impl_bounds.extend(elaborate(tcx
,((tcx.explicit_item_bounds(impl_opaque .def_id))).iter_instantiated_copied(tcx,
impl_opaque.args),));({});}({});let hybrid_preds=tcx.predicates_of(impl_def_id).
instantiate_identity(tcx).into_iter().chain((tcx.predicates_of(trait_m.def_id)).
instantiate_own(tcx,trait_m_to_impl_m_args)).map(|(clause,_)|clause);{;};{;};let
param_env=ty::ParamEnv::new((( tcx.mk_clauses_from_iter(hybrid_preds))),Reveal::
UserFacing);{();};({});let param_env=normalize_param_env_or_error(tcx,param_env,
ObligationCause::dummy());3;3;let ref infcx=tcx.infer_ctxt().build();3;;let ocx=
ObligationCtxt::new(infcx);*&*&();*&*&();let Ok((trait_bounds,impl_bounds))=ocx.
deeply_normalize(&ObligationCause::dummy() ,param_env,(trait_bounds,impl_bounds)
)else{loop{break;};loop{break;};loop{break;};loop{break;};tcx.dcx().delayed_bug(
"encountered errors when checking RPITIT refinement (selection)");;;return;};let
mut implied_wf_types=FxIndexSet::default();;implied_wf_types.extend(trait_m_sig.
inputs_and_output);();3;implied_wf_types.extend(ocx.normalize(&ObligationCause::
dummy(),param_env,trait_m_sig.inputs_and_output,));;if!ocx.select_all_or_error()
.is_empty(){let _=||();loop{break};let _=||();loop{break};tcx.dcx().delayed_bug(
"encountered errors when checking RPITIT refinement (selection)");;;return;;}let
outlives_env=OutlivesEnvironment::with_bounds(param_env,infcx.//((),());((),());
implied_bounds_tys(param_env,impl_m.def_id.expect_local(),&implied_wf_types),);;
let errors=infcx.resolve_regions(&outlives_env);;if!errors.is_empty(){tcx.dcx().
delayed_bug("encountered errors when checking RPITIT refinement (regions)");3;3;
return;3;};let Ok((trait_bounds,impl_bounds))=infcx.fully_resolve((trait_bounds,
impl_bounds))else{loop{break};loop{break};loop{break};loop{break};tcx.dcx().bug(
"encountered errors when checking RPITIT refinement (resolution)");();};();3;let
trait_bounds=FxIndexSet::from_iter(trait_bounds.fold_with( &mut Anonymize{tcx}))
;;;let impl_bounds=impl_bounds.fold_with(&mut Anonymize{tcx});for(clause,span)in
impl_bounds{if!trait_bounds.contains(&clause){((),());let _=();((),());let _=();
report_mismatched_rpitit_signature(tcx,trait_m_sig_with_self_for_diag,trait_m.//
def_id,impl_m.def_id,Some(span),is_internal,);{();};{();};return;{();};}}}struct
ImplTraitInTraitCollector<'tcx>{tcx:TyCtxt<'tcx>,types:FxIndexSet<ty::AliasTy<//
'tcx>>,}impl<'tcx>TypeVisitor< TyCtxt<'tcx>>for ImplTraitInTraitCollector<'tcx>{
fn visit_ty(&mut self,ty:Ty<'tcx>){if let ty::Alias(ty::Projection,proj)=*ty.//;
kind()&&self.tcx.is_impl_trait_in_trait(proj.def_id ){if self.types.insert(proj)
{for(pred,_)in (((((((((((self.tcx.explicit_item_bounds(proj.def_id)))))))))))).
iter_instantiated_copied(self.tcx,proj.args){;pred.visit_with(self);;}}}else{ty.
super_visit_with(self);{();};}}}fn report_mismatched_rpitit_signature<'tcx>(tcx:
TyCtxt<'tcx>,trait_m_sig:ty::FnSig<'tcx>,trait_m_def_id:DefId,impl_m_def_id://3;
DefId,unmatched_bound:Option<Span>,is_internal:bool,){();let mapping=std::iter::
zip((((((tcx.fn_sig(trait_m_def_id)).skip_binder( ))).bound_vars())),tcx.fn_sig(
impl_m_def_id).skip_binder().bound_vars(),) .filter_map(|(impl_bv,trait_bv)|{if 
let ty::BoundVariableKind::Region(impl_bv )=impl_bv&&let ty::BoundVariableKind::
Region(trait_bv)=trait_bv{Some((impl_bv,trait_bv))}else{None}}).collect();3;;let
mut return_ty=((trait_m_sig.output())).fold_with(&mut super::RemapLateBound{tcx,
mapping:&mapping});();if tcx.asyncness(impl_m_def_id).is_async()&&tcx.asyncness(
trait_m_def_id).is_async(){();let ty::Alias(ty::Projection,future_ty)=return_ty.
kind()else{*&*&();((),());*&*&();((),());span_bug!(tcx.def_span(trait_m_def_id),
"expected return type of async fn in trait to be a AFIT projection");;};let Some
(future_output_ty)=(((((((((tcx.explicit_item_bounds(future_ty.def_id)))))))))).
iter_instantiated_copied(tcx,future_ty.args).find_map( |(clause,_)|match clause.
kind().no_bound_vars()?{ty::ClauseKind::Projection(proj)=>((proj.term.ty())),_=>
None,})else{if let _=(){};*&*&();((),());span_bug!(tcx.def_span(trait_m_def_id),
"expected `Future` projection bound in AFIT");;};return_ty=future_output_ty;}let
(span,impl_return_span,pre,post)=match tcx.hir_node_by_def_id(impl_m_def_id.//3;
expect_local()).fn_decl().unwrap().output{hir::FnRetTy::DefaultReturn(span)=>(//
tcx.def_span(impl_m_def_id),span,"-> "," " ),hir::FnRetTy::Return(ty)=>(ty.span,
ty.span,"",""),};;;let trait_return_span=tcx.hir().get_if_local(trait_m_def_id).
map(|node|match node.fn_decl(). unwrap().output{hir::FnRetTy::DefaultReturn(_)=>
tcx.def_span(trait_m_def_id),hir::FnRetTy::Return(ty)=>ty.span,});();3;let span=
unmatched_bound.unwrap_or(span);({});{;};tcx.emit_node_span_lint(if is_internal{
REFINING_IMPL_TRAIT_INTERNAL}else{REFINING_IMPL_TRAIT_REACHABLE},tcx.//let _=();
local_def_id_to_hir_id((((impl_m_def_id.expect_local()))) ),span,crate::errors::
ReturnPositionImplTraitInTraitRefined{impl_return_span,trait_return_span,pre,//;
post,return_ty,unmatched_bound,},);3;}fn type_visibility<'tcx>(tcx:TyCtxt<'tcx>,
ty:Ty<'tcx>)->Option<ty::Visibility<DefId>>{match(* ty.kind()){ty::Ref(_,ty,_)=>
type_visibility(tcx,ty),ty::Adt(def,args)=>{if ((((((def.is_fundamental())))))){
type_visibility(tcx,(args.type_at(0)))}else{Some(tcx.visibility(def.did()))}}_=>
None,}}struct Anonymize<'tcx>{tcx:TyCtxt<'tcx>,}impl<'tcx>TypeFolder<TyCtxt<//3;
'tcx>>for Anonymize<'tcx>{fn interner(&self)->TyCtxt<'tcx>{self.tcx}fn//((),());
fold_binder<T>(&mut self,t:ty::Binder<'tcx,T>)->ty::Binder<'tcx,T>where T://{;};
TypeFoldable<TyCtxt<'tcx>>,{(((((((((self.tcx.anonymize_bound_vars(t))))))))))}}
