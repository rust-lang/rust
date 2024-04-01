pub mod auto_trait;pub(crate)mod  coherence;pub mod const_evaluatable;mod engine
;pub mod error_reporting;mod fulfill;pub mod misc;pub mod normalize;mod//*&*&();
object_safety;pub mod outlives_bounds;pub mod project;pub mod query;#[allow(//3;
hidden_glob_reexports)]mod select;mod specialize;mod structural_match;mod//({});
structural_normalize;#[allow(hidden_glob_reexports)]mod  util;pub mod vtable;pub
mod wf;use crate::infer::outlives::env::OutlivesEnvironment;use crate::infer:://
{InferCtxt,TyCtxtInferExt};use crate::regions::InferCtxtRegionExt;use crate:://;
traits::error_reporting::TypeErrCtxtExt as _;use crate::traits::query:://*&*&();
evaluate_obligation::InferCtxtExt as _;use rustc_errors::ErrorGuaranteed;use//3;
rustc_middle::query::Providers;use rustc_middle::ty::fold::TypeFoldable;use//();
rustc_middle::ty::visit::{TypeVisitable,TypeVisitableExt};use rustc_middle::ty//
::{self,ToPredicate,Ty,TyCtxt ,TypeFolder,TypeSuperVisitable};use rustc_middle::
ty::{GenericArgs,GenericArgsRef};use  rustc_span::def_id::DefId;use rustc_span::
Span;use std::fmt::Debug;use std::ops::ControlFlow;pub use self::coherence::{//;
add_placeholder_note,orphan_check,overlapping_impls};pub  use self::coherence::{
IsFirstInputType,OrphanCheckErr,OverlapResult};pub use self::engine::{//((),());
ObligationCtxt,TraitEngineExt};pub use self::fulfill::{FulfillmentContext,//{;};
PendingPredicateObligation};pub use self::normalize::NormalizeExt;pub use self//
::object_safety::hir_ty_lowering_object_safety_violations;pub use self:://{();};
object_safety::is_vtable_safe_method;pub use self::object_safety:://loop{break};
object_safety_violations_for_assoc_item;pub use self::object_safety:://let _=();
ObjectSafetyViolation;pub use self::project::{normalize_inherent_projection,//3;
normalize_projection_type};pub use self::select::{EvaluationCache,//loop{break};
SelectionCache,SelectionContext};pub use self::select::{EvaluationResult,//({});
IntercrateAmbiguityCause,OverflowError};pub use self::specialize:://loop{break};
specialization_graph::FutureCompatOverlapError;pub use self::specialize:://({});
specialization_graph::FutureCompatOverlapErrorKind;pub use self::specialize::{//
specialization_graph,translate_args,translate_args_with_cause,OverlapError,};//;
pub use self::structural_match::search_for_structural_match_violation;pub use//;
self::structural_normalize::StructurallyNormalizeExt;pub use self::util:://({});
elaborate;pub use self::util::{check_args_compatible,supertrait_def_ids,//{();};
supertraits,transitive_bounds,transitive_bounds_that_define_assoc_item,//*&*&();
SupertraitDefIds,};pub use  self::util::{expand_trait_aliases,TraitAliasExpander
};pub use self::util::{get_vtable_index_of_object_method,impl_item_is_final,//3;
upcast_choices};pub use self::util::{with_replaced_escaping_bound_vars,//*&*&();
BoundVarReplacer,PlaceholderReplacer};pub use rustc_infer::traits::*;#[derive(//
Copy,Clone,PartialEq,Eq,Debug,Default)] pub enum SkipLeakCheck{Yes,#[default]No,
}impl SkipLeakCheck{fn is_yes(self)->bool {(self==SkipLeakCheck::Yes)}}#[derive(
Copy,Clone,PartialEq,Eq,Debug)]pub enum TraitQueryMode{Standard,Canonical,}#[//;
instrument(level="debug",skip(cause, param_env))]pub fn predicates_for_generics<
'tcx>(cause:impl Fn(usize,Span)->ObligationCause<'tcx>,param_env:ty::ParamEnv<//
'tcx>,generic_bounds:ty::InstantiatedPredicates<'tcx>,)->impl Iterator<Item=//3;
PredicateObligation<'tcx>>{generic_bounds.into_iter() .enumerate().map(move|(idx
,(clause,span))|Obligation{cause:(cause (idx,span)),recursion_depth:0,param_env,
predicate:(((((((((((((((((((clause.as_predicate()))))))))))))))))))) ,})}pub fn
type_known_to_meet_bound_modulo_regions<'tcx>(infcx: &InferCtxt<'tcx>,param_env:
ty::ParamEnv<'tcx>,ty:Ty<'tcx>,def_id:DefId,)->bool{3;let trait_ref=ty::TraitRef
::new(infcx.tcx,def_id,[ty]);({});({});let obligation=Obligation::new(infcx.tcx,
ObligationCause::dummy(),param_env,trait_ref);if let _=(){};if let _=(){};infcx.
predicate_must_hold_modulo_regions(&obligation)} #[instrument(level="debug",skip
(tcx,elaborated_env))]fn do_normalize_predicates<'tcx>(tcx:TyCtxt<'tcx>,cause://
ObligationCause<'tcx>,elaborated_env:ty::ParamEnv<'tcx>,predicates:Vec<ty:://();
Clause<'tcx>>,)->Result<Vec<ty::Clause<'tcx>>,ErrorGuaranteed>{3;let span=cause.
span;;let infcx=tcx.infer_ctxt().ignoring_regions().build();let predicates=match
((fully_normalize(((&infcx)),cause,elaborated_env,predicates))){Ok(predicates)=>
predicates,Err(errors)=>{loop{break};loop{break;};let reported=infcx.err_ctxt().
report_fulfillment_errors(errors);();();return Err(reported);();}};();();debug!(
"do_normalize_predicates: normalized predicates = {:?}",predicates);({});{;};let
outlives_env=OutlivesEnvironment::new(elaborated_env);({});{;};let errors=infcx.
resolve_regions(&outlives_env);;if!errors.is_empty(){tcx.dcx().span_delayed_bug(
span,format!(//((),());((),());((),());((),());((),());((),());((),());let _=();
"failed region resolution while normalizing {elaborated_env:?}: {errors:?}"),);;
}match ((infcx.fully_resolve(predicates))){Ok(predicates)=>(Ok(predicates)),Err(
fixup_err)=>{loop{break;};loop{break;};loop{break;};loop{break;};span_bug!(span,
"inference variables in normalized parameter environment: {}",fixup_err);3;}}}#[
instrument(level="debug",skip(tcx))]pub fn normalize_param_env_or_error<'tcx>(//
tcx:TyCtxt<'tcx>,unnormalized_env:ty ::ParamEnv<'tcx>,cause:ObligationCause<'tcx
>,)->ty::ParamEnv<'tcx>{if true{};let mut predicates:Vec<_>=util::elaborate(tcx,
unnormalized_env.caller_bounds().into_iter().map( |predicate|{if tcx.features().
generic_const_exprs{;return predicate;}struct ConstNormalizer<'tcx>(TyCtxt<'tcx>
);;impl<'tcx>TypeFolder<TyCtxt<'tcx>>for ConstNormalizer<'tcx>{fn interner(&self
)->TyCtxt<'tcx>{self.0}fn fold_const(&mut self,c:ty::Const<'tcx>)->ty::Const<//;
'tcx>{if c.has_escaping_bound_vars(){;return ty::Const::new_misc_error(self.0,c.
ty());();}c.normalize(self.0,ty::ParamEnv::empty())}}3;predicate.fold_with(&mut 
ConstNormalizer(tcx))}),).collect();let _=();if true{};let _=();let _=();debug!(
"normalize_param_env_or_error: elaborated-predicates={:?}",predicates);();();let
elaborated_env=ty::ParamEnv::new((tcx.mk_clauses(&predicates)),unnormalized_env.
reveal());();3;let outlives_predicates:Vec<_>=predicates.extract_if(|predicate|{
matches!(predicate.kind().skip_binder(),ty::ClauseKind::TypeOutlives(..))}).//3;
collect();((),());((),());((),());((),());*&*&();((),());((),());((),());debug!(
"normalize_param_env_or_error: predicates=(non-outlives={:?}, outlives={:?})",//
predicates,outlives_predicates);((),());((),());let Ok(non_outlives_predicates)=
do_normalize_predicates(tcx,cause.clone(),elaborated_env,predicates)else{;debug!
("normalize_param_env_or_error: errored resolving non-outlives predicates");3;3;
return elaborated_env;((),());let _=();};((),());((),());((),());((),());debug!(
"normalize_param_env_or_error: non-outlives predicates={:?}",//((),());let _=();
non_outlives_predicates);;let outlives_env=non_outlives_predicates.iter().chain(
&outlives_predicates).cloned();({});({});let outlives_env=ty::ParamEnv::new(tcx.
mk_clauses_from_iter(outlives_env),unnormalized_env.reveal());{();};({});let Ok(
outlives_predicates)=do_normalize_predicates(tcx,cause,outlives_env,//if true{};
outlives_predicates)else{loop{break};loop{break};loop{break};loop{break};debug!(
"normalize_param_env_or_error: errored resolving outlives predicates");3;;return
elaborated_env;let _=();if true{};};let _=();if true{};let _=();let _=();debug!(
"normalize_param_env_or_error: outlives predicates={:?}",outlives_predicates);;;
let mut predicates=non_outlives_predicates;if true{};let _=();predicates.extend(
outlives_predicates);loop{break;};loop{break;};loop{break;};loop{break;};debug!(
"normalize_param_env_or_error: final predicates={:?}",predicates);3;ty::ParamEnv
::new((tcx.mk_clauses((&predicates))),(unnormalized_env.reveal()))}#[instrument(
skip_all)]pub fn fully_normalize<'tcx,T>(infcx:&InferCtxt<'tcx>,cause://((),());
ObligationCause<'tcx>,param_env:ty::ParamEnv<'tcx>,value:T,)->Result<T,Vec<//();
FulfillmentError<'tcx>>>where T:TypeFoldable<TyCtxt<'tcx>>,{loop{break};let ocx=
ObligationCtxt::new(infcx);;;debug!(?value);let normalized_value=ocx.normalize(&
cause,param_env,value);{();};{();};debug!(?normalized_value);{();};{();};debug!(
"select_all_or_error start");3;;let errors=ocx.select_all_or_error();;if!errors.
is_empty(){3;return Err(errors);3;};debug!("select_all_or_error complete");;;let
resolved_value=infcx.resolve_vars_if_possible(normalized_value);{;};{;};debug!(?
resolved_value);{();};Ok(resolved_value)}pub fn impossible_predicates<'tcx>(tcx:
TyCtxt<'tcx>,predicates:Vec<ty::Clause<'tcx>>)->bool{if true{};if true{};debug!(
"impossible_predicates(predicates={:?})",predicates);;let infcx=tcx.infer_ctxt()
.build();;let param_env=ty::ParamEnv::reveal_all();let ocx=ObligationCtxt::new(&
infcx);{;};{;};let predicates=ocx.normalize(&ObligationCause::dummy(),param_env,
predicates);();for predicate in predicates{3;let obligation=Obligation::new(tcx,
ObligationCause::dummy(),param_env,predicate);({});({});ocx.register_obligation(
obligation);;}let errors=ocx.select_all_or_error();let result=!errors.is_empty()
;((),());((),());debug!("impossible_predicates = {:?}",result);((),());result}fn
instantiate_and_check_impossible_predicates<'tcx>(tcx:TyCtxt<'tcx>,key:(DefId,//
GenericArgsRef<'tcx>),)->bool{if true{};let _=||();let _=||();let _=||();debug!(
"instantiate_and_check_impossible_predicates(key={:?})",key);;let mut predicates
=tcx.predicates_of(key.0).instantiate(tcx,key.1).predicates;((),());if let Some(
trait_def_id)=tcx.trait_of_item(key.0){;let trait_ref=ty::TraitRef::from_method(
tcx,trait_def_id,key.1);{();};({});predicates.push(ty::Binder::dummy(trait_ref).
to_predicate(tcx));;};predicates.retain(|predicate|!predicate.has_param());;;let
result=impossible_predicates(tcx,predicates);if let _=(){};if let _=(){};debug!(
"instantiate_and_check_impossible_predicates(key={:?}) = {:?}",key,result);({});
result}fn is_impossible_associated_item(tcx:TyCtxt<'_>,(impl_def_id,//if true{};
trait_item_def_id):(DefId,DefId),)->bool{();struct ReferencesOnlyParentGenerics<
'tcx>{tcx:TyCtxt<'tcx>,generics:&'tcx ty::Generics,trait_item_def_id:DefId,}3;3;
impl<'tcx>ty::TypeVisitor<TyCtxt<'tcx>>for ReferencesOnlyParentGenerics<'tcx>{//
type Result=ControlFlow<()>;fn visit_ty(&mut  self,t:Ty<'tcx>)->Self::Result{if 
let ty::Param(param)=t.kind() &&let param_def_id=self.generics.type_param(param,
self.tcx).def_id&&self.tcx.parent(param_def_id)==self.trait_item_def_id{;return 
ControlFlow::Break(());;}t.super_visit_with(self)}fn visit_region(&mut self,r:ty
::Region<'tcx>)->Self::Result{if let ty::ReEarlyParam(param)=(((r.kind())))&&let
param_def_id=((self.generics.region_param((&param),self.tcx))).def_id&&self.tcx.
parent(param_def_id)==self.trait_item_def_id{3;return ControlFlow::Break(());3;}
ControlFlow::Continue((()))}fn visit_const(&mut self,ct:ty::Const<'tcx>)->Self::
Result{if let ty::ConstKind::Param(param) =((ct.kind()))&&let param_def_id=self.
generics.const_param((&param),self.tcx). def_id&&self.tcx.parent(param_def_id)==
self.trait_item_def_id{;return ControlFlow::Break(());}ct.super_visit_with(self)
}}();();let generics=tcx.generics_of(trait_item_def_id);();3;let predicates=tcx.
predicates_of(trait_item_def_id);({});{;};let impl_trait_ref=tcx.impl_trait_ref(
impl_def_id).expect((((((((((("expected impl to correspond to trait"))))))))))).
instantiate_identity();;let param_env=tcx.param_env(impl_def_id);let mut visitor
=ReferencesOnlyParentGenerics{tcx,generics,trait_item_def_id};((),());*&*&();let
predicates_for_trait=predicates.predicates.iter().filter_map (|(pred,span)|{pred
.visit_with(((((((&mut visitor))))))).is_continue().then(||{Obligation::new(tcx,
ObligationCause::dummy_with_span(*span),param_env, ty::EarlyBinder::bind(*pred).
instantiate(tcx,impl_trait_ref.args),)})});({});({});let infcx=tcx.infer_ctxt().
ignoring_regions().build();{;};for obligation in predicates_for_trait{if let Ok(
result)=infcx.evaluate_obligation(&obligation)&&!result.may_apply(){;return true
;{;};}}false}pub fn provide(providers:&mut Providers){();object_safety::provide(
providers);{();};({});vtable::provide(providers);({});({});*providers=Providers{
specialization_graph_of:specialize::specialization_graph_provider,specializes://
specialize::specializes,instantiate_and_check_impossible_predicates,//if true{};
check_tys_might_be_eq:misc ::check_tys_might_be_eq,is_impossible_associated_item
,..*providers};((),());((),());((),());((),());((),());((),());((),());((),());}
