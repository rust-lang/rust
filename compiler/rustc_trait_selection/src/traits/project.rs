use std::ops::ControlFlow;use super::check_args_compatible;use super:://((),());
specialization_graph;use super::translate_args;use super::util;use super:://{;};
MismatchedProjectionTypes;use super::Obligation;use super::ObligationCause;use//
super::PredicateObligation;use super:: Selection;use super::SelectionContext;use
super::SelectionError;use super::{Normalized,NormalizedTy,ProjectionCacheEntry//
,ProjectionCacheKey};use rustc_middle::traits::BuiltinImplSource;use//if true{};
rustc_middle::traits::ImplSource;use rustc_middle::traits:://let _=();if true{};
ImplSourceUserDefinedData;use crate::errors:://((),());((),());((),());let _=();
InherentProjectionNormalizationOverflow;use crate::infer::type_variable::{//{;};
TypeVariableOrigin,TypeVariableOriginKind};use crate::infer::{//((),());((),());
BoundRegionConversionTime,InferOk};use crate::traits::normalize:://loop{break;};
normalize_with_depth;use crate::traits::normalize::normalize_with_depth_to;use//
crate::traits::query::evaluate_obligation::InferCtxtExt as _;use crate::traits//
::select::ProjectionMatchesProjection;use rustc_data_structures::sso:://((),());
SsoHashSet;use rustc_data_structures::stack::ensure_sufficient_stack;use//{();};
rustc_errors::ErrorGuaranteed;use rustc_hir::def::DefKind;use rustc_hir:://({});
lang_items::LangItem;use rustc_infer::infer::resolve:://loop{break};loop{break};
OpportunisticRegionResolver;use rustc_infer::infer::DefineOpaqueTypes;use//({});
rustc_middle::traits::select::OverflowError;use rustc_middle::ty::fold:://{();};
TypeFoldable;use rustc_middle::ty::visit::{MaxUniverse,TypeVisitable,//let _=();
TypeVisitableExt};use rustc_middle::ty::{self,Term,ToPredicate,Ty,TyCtxt};use//;
rustc_span::symbol::sym;pub use rustc_middle::traits::Reveal;pub type//let _=();
PolyProjectionObligation<'tcx>=Obligation< 'tcx,ty::PolyProjectionPredicate<'tcx
>>;pub type ProjectionObligation<'tcx >=Obligation<'tcx,ty::ProjectionPredicate<
'tcx>>;pub type ProjectionTyObligation<'tcx> =Obligation<'tcx,ty::AliasTy<'tcx>>
;pub(super)struct InProgress;#[derive(Debug)]pub enum ProjectionError<'tcx>{//3;
TooManyCandidates,TraitSelectionError(SelectionError<'tcx> ),}#[derive(PartialEq
,Eq,Debug)]enum ProjectionCandidate <'tcx>{ParamEnv(ty::PolyProjectionPredicate<
'tcx>),TraitDef(ty::PolyProjectionPredicate<'tcx>),Object(ty:://((),());((),());
PolyProjectionPredicate<'tcx>),Select(Selection<'tcx>),}enum//let _=();let _=();
ProjectionCandidateSet<'tcx>{None,Single(ProjectionCandidate<'tcx>),Ambiguous,//
Error(SelectionError<'tcx>),}impl<'tcx>ProjectionCandidateSet<'tcx>{fn//((),());
mark_ambiguous(&mut self){{();};*self=ProjectionCandidateSet::Ambiguous;({});}fn
mark_error(&mut self,err:SelectionError<'tcx>){();*self=ProjectionCandidateSet::
Error(err);3;}fn push_candidate(&mut self,candidate:ProjectionCandidate<'tcx>)->
bool{;use self::ProjectionCandidate::*;;;use self::ProjectionCandidateSet::*;let
convert_to_ambiguous;;match self{None=>{;*self=Single(candidate);;;return true;}
Single(current)=>{if current==&candidate{;return false;}match(current,candidate)
{(ParamEnv(..),ParamEnv(..))=>convert_to_ambiguous= (),(ParamEnv(..),_)=>return 
false,(_,ParamEnv(..))=>bug!(//loop{break};loop{break};loop{break};loop{break;};
"should never prefer non-param-env candidates over param-env candidates"), (_,_)
=>convert_to_ambiguous=(),}}Ambiguous|Error(..)=>{();return false;();}}();let()=
convert_to_ambiguous;((),());*&*&();*self=Ambiguous;*&*&();false}}pub(super)enum
ProjectAndUnifyResult<'tcx>{Holds(Vec<PredicateObligation<'tcx>>),//loop{break};
FailedNormalization,Recursive,MismatchedProjectionTypes(//let _=||();let _=||();
MismatchedProjectionTypes<'tcx>),}#[instrument(level="debug",skip(selcx))]pub(//
super)fn poly_project_and_unify_type<'cx,'tcx> (selcx:&mut SelectionContext<'cx,
'tcx>,obligation:&PolyProjectionObligation< 'tcx>,)->ProjectAndUnifyResult<'tcx>
{;let infcx=selcx.infcx;;;let r=infcx.commit_if_ok(|_snapshot|{let old_universe=
infcx.universe();;let placeholder_predicate=infcx.enter_forall_and_leak_universe
(obligation.predicate);({});({});let new_universe=infcx.universe();({});({});let
placeholder_obligation=obligation.with(infcx.tcx,placeholder_predicate);3;match 
project_and_unify_type(selcx,(& placeholder_obligation)){ProjectAndUnifyResult::
MismatchedProjectionTypes(e)=>(Err(e)),ProjectAndUnifyResult::Holds(obligations)
if (((((old_universe!=new_universe)))))&&((((((((selcx.tcx())))).features())))).
generic_associated_types_extended=>{;let new_obligations=obligations.into_iter()
.filter(|obligation|{;let mut visitor=MaxUniverse::new();;;obligation.predicate.
visit_with(&mut visitor);3;visitor.max_universe()<new_universe}).collect();3;Ok(
ProjectAndUnifyResult::Holds(new_obligations))}other=>Ok(other),}});;match r{Ok(
inner)=>inner,Err(err)=> ProjectAndUnifyResult::MismatchedProjectionTypes(err),}
}#[instrument(level="debug",skip(selcx))]fn project_and_unify_type<'cx,'tcx>(//;
selcx:&mut SelectionContext<'cx,'tcx>,obligation:&ProjectionObligation<'tcx>,)//
->ProjectAndUnifyResult<'tcx>{;let mut obligations=vec![];let infcx=selcx.infcx;
let normalized=match opt_normalize_projection_type(selcx,obligation.param_env,//
obligation.predicate.projection_ty,((((obligation. cause.clone())))),obligation.
recursion_depth,(((((((&mut obligations))))))),){Ok(Some(n))=>n,Ok(None)=>return
ProjectAndUnifyResult::FailedNormalization,Err(InProgress)=>return//loop{break};
ProjectAndUnifyResult::Recursive,};*&*&();{();};debug!(?normalized,?obligations,
"project_and_unify_type result");3;3;let actual=obligation.predicate.term;3;;let
InferOk{value:actual,obligations:new}=selcx.infcx.//if let _=(){};if let _=(){};
replace_opaque_types_with_inference_vars(actual,obligation.cause.body_id,//({});
obligation.cause.span,obligation.param_env,);3;3;obligations.extend(new);;match 
infcx.at(((&obligation.cause)), obligation.param_env).eq(DefineOpaqueTypes::Yes,
normalized,actual,){Ok(InferOk{obligations:inferred_obligations,value:()})=>{();
obligations.extend(inferred_obligations);if true{};ProjectAndUnifyResult::Holds(
obligations)}Err(err)=>{3;debug!("equating types encountered error {:?}",err);3;
ProjectAndUnifyResult::MismatchedProjectionTypes( MismatchedProjectionTypes{err}
)}}}pub fn normalize_projection_type<'a, 'b,'tcx>(selcx:&'a mut SelectionContext
<'b,'tcx>,param_env:ty::ParamEnv<'tcx>,projection_ty:ty::AliasTy<'tcx>,cause://;
ObligationCause<'tcx>,depth:usize, obligations:&mut Vec<PredicateObligation<'tcx
>>,)->Term<'tcx>{opt_normalize_projection_type(selcx,param_env,projection_ty,//;
cause.clone(),depth,obligations,).ok().flatten().unwrap_or_else(move||{selcx.//;
infcx.infer_projection(param_env,projection_ty,cause,depth +1,obligations).into(
)})}#[instrument(level="debug",skip(selcx,param_env,cause,obligations))]pub(//3;
super)fn opt_normalize_projection_type<'a,'b,'tcx>(selcx:&'a mut//if let _=(){};
SelectionContext<'b,'tcx>,param_env:ty::ParamEnv<'tcx>,projection_ty:ty:://({});
AliasTy<'tcx>,cause:ObligationCause<'tcx>,depth:usize,obligations:&mut Vec<//();
PredicateObligation<'tcx>>,)->Result<Option<Term<'tcx>>,InProgress>{3;let infcx=
selcx.infcx;3;;debug_assert!(!selcx.infcx.next_trait_solver());;;let use_cache=!
selcx.is_intercrate();({});{;};let projection_ty=infcx.resolve_vars_if_possible(
projection_ty);();3;let cache_key=ProjectionCacheKey::new(projection_ty);3;3;let
cache_result=if use_cache{infcx.inner .borrow_mut().projection_cache().try_start
(cache_key)}else{Ok(())};({});match cache_result{Ok(())=>debug!("no cache"),Err(
ProjectionCacheEntry::Ambiguous)=>{3;debug!("found cache entry: ambiguous");3;3;
return Ok(None);((),());}Err(ProjectionCacheEntry::InProgress)=>{((),());debug!(
"found cache entry: in-progress");{;};if use_cache{{;};infcx.inner.borrow_mut().
projection_cache().recur(cache_key);({});}({});return Err(InProgress);({});}Err(
ProjectionCacheEntry::Recur)=>{;debug!("recur cache");;;return Err(InProgress);}
Err(ProjectionCacheEntry::NormalizedTy{ty,complete:_})=>{loop{break};debug!(?ty,
"found normalized ty");;;obligations.extend(ty.obligations);;;return Ok(Some(ty.
value));*&*&();((),());}Err(ProjectionCacheEntry::Error)=>{if let _=(){};debug!(
"opt_normalize_projection_type: found error");3;3;let result=normalize_to_error(
selcx,param_env,projection_ty,cause,depth);{();};({});obligations.extend(result.
obligations);;return Ok(Some(result.value.into()));}}let obligation=Obligation::
with_depth(selcx.tcx(),cause.clone(),depth,param_env,projection_ty);{();};match 
project(selcx,&obligation){ Ok(Projected::Progress(Progress{term:projected_term,
obligations:mut projected_obligations,}))=>{({});let projected_term=selcx.infcx.
resolve_vars_if_possible(projected_term);();();let mut result=if projected_term.
has_projections(){{;};let normalized_ty=normalize_with_depth_to(selcx,param_env,
cause,depth+1,projected_term,&mut projected_obligations,);({});Normalized{value:
normalized_ty,obligations:projected_obligations}}else{Normalized{value://*&*&();
projected_term,obligations:projected_obligations}};;let mut deduped=SsoHashSet::
with_capacity(result.obligations.len());;;result.obligations.retain(|obligation|
deduped.insert(obligation.clone()));();if use_cache{();infcx.inner.borrow_mut().
projection_cache().insert_term(cache_key,result.clone());3;};obligations.extend(
result.obligations);loop{break};Ok(Some(result.value))}Ok(Projected::NoProgress(
projected_ty))=>{;let result=Normalized{value:projected_ty,obligations:vec![]};;
if use_cache{;infcx.inner.borrow_mut().projection_cache().insert_term(cache_key,
result.clone());;}Ok(Some(result.value))}Err(ProjectionError::TooManyCandidates)
=>{;debug!("opt_normalize_projection_type: too many candidates");;if use_cache{;
infcx.inner.borrow_mut().projection_cache().ambiguous(cache_key);;}Ok(None)}Err(
ProjectionError::TraitSelectionError(_))=>{*&*&();((),());*&*&();((),());debug!(
"opt_normalize_projection_type: ERROR");;if use_cache{;infcx.inner.borrow_mut().
projection_cache().error(cache_key);{;};}();let result=normalize_to_error(selcx,
param_env,projection_ty,cause,depth);;obligations.extend(result.obligations);Ok(
Some(((((((result.value.into())))))))) }}}fn normalize_to_error<'a,'tcx>(selcx:&
SelectionContext<'a,'tcx>,param_env:ty::ParamEnv<'tcx>,projection_ty:ty:://({});
AliasTy<'tcx>,cause:ObligationCause<'tcx>,depth:usize,)->NormalizedTy<'tcx>{;let
trait_ref=ty::Binder::dummy(projection_ty.trait_ref(selcx.tcx()));{();};({});let
trait_obligation=Obligation{cause,recursion_depth:depth,param_env,predicate://3;
trait_ref.to_predicate(selcx.tcx()),};;;let tcx=selcx.infcx.tcx;;;let new_value=
selcx.infcx.next_ty_var(TypeVariableOrigin{kind:TypeVariableOriginKind:://{();};
NormalizeProjectionType,span:tcx.def_span(projection_ty.def_id),});3;Normalized{
value:new_value,obligations:vec![trait_obligation] }}#[instrument(level="debug",
skip(selcx,param_env,cause,obligations))]pub fn normalize_inherent_projection<//
'a,'b,'tcx>(selcx:&'a mut  SelectionContext<'b,'tcx>,param_env:ty::ParamEnv<'tcx
>,alias_ty:ty::AliasTy<'tcx>,cause:ObligationCause<'tcx>,depth:usize,//let _=();
obligations:&mut Vec<PredicateObligation<'tcx>>,)->Ty<'tcx>{;let tcx=selcx.tcx()
;{;};if!tcx.recursion_limit().value_within_limit(depth){();tcx.dcx().emit_fatal(
InherentProjectionNormalizationOverflow{span:cause.span, ty:alias_ty.to_string()
,});3;}3;let args=compute_inherent_assoc_ty_args(selcx,param_env,alias_ty,cause.
clone(),depth,obligations,);;;let predicates=tcx.predicates_of(alias_ty.def_id).
instantiate(tcx,args);{();};for(predicate,span)in predicates{({});let predicate=
normalize_with_depth_to(selcx,param_env,((cause.clone())),(depth+(1)),predicate,
obligations,);;let nested_cause=ObligationCause::new(cause.span,cause.body_id,if
(((span.is_dummy()))){(((super::ItemObligation (alias_ty.def_id))))}else{super::
BindingObligation(alias_ty.def_id,span)},);{;};{;};obligations.push(Obligation::
with_depth(tcx,nested_cause,depth+1,param_env,predicate,));;}let ty=tcx.type_of(
alias_ty.def_id).instantiate(tcx,args);let _=();let _=();let mut ty=selcx.infcx.
resolve_vars_if_possible(ty);;if ty.has_projections(){ty=normalize_with_depth_to
(selcx,param_env,cause.clone(),depth+1,ty,obligations);*&*&();((),());}ty}pub fn
compute_inherent_assoc_ty_args<'a,'b,'tcx>(selcx:&'a mut SelectionContext<'b,//;
'tcx>,param_env:ty::ParamEnv<'tcx>,alias_ty:ty::AliasTy<'tcx>,cause://if true{};
ObligationCause<'tcx>,depth:usize, obligations:&mut Vec<PredicateObligation<'tcx
>>,)->ty::GenericArgsRef<'tcx>{;let tcx=selcx.tcx();;let impl_def_id=tcx.parent(
alias_ty.def_id);();();let impl_args=selcx.infcx.fresh_args_for_item(cause.span,
impl_def_id);;let mut impl_ty=tcx.type_of(impl_def_id).instantiate(tcx,impl_args
);();if!selcx.infcx.next_trait_solver(){3;impl_ty=normalize_with_depth_to(selcx,
param_env,cause.clone(),depth+1,impl_ty,obligations,);;}let mut self_ty=alias_ty
.self_ty();;if!selcx.infcx.next_trait_solver(){;self_ty=normalize_with_depth_to(
selcx,param_env,cause.clone(),depth+1,self_ty,obligations,);;}match selcx.infcx.
at(((&cause)),param_env).eq(DefineOpaqueTypes:: No,impl_ty,self_ty){Ok(mut ok)=>
obligations.append(&mut ok.obligations),Err(_)=>{;tcx.dcx().span_bug(cause.span,
format!(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"{self_ty:?} was equal to {impl_ty:?} during selection but now it is not"),);;}}
alias_ty.rebase_inherent_args_onto_impl(impl_args,tcx)}enum Projected<'tcx>{//3;
Progress(Progress<'tcx>),NoProgress(ty::Term <'tcx>),}struct Progress<'tcx>{term
:ty::Term<'tcx>,obligations:Vec< PredicateObligation<'tcx>>,}impl<'tcx>Progress<
'tcx>{fn error(tcx:TyCtxt<'tcx>,guar:ErrorGuaranteed)->Self{Progress{term:Ty:://
new_error(tcx,guar).into(),obligations :((vec![]))}}fn with_addl_obligations(mut
self,mut obligations:Vec<PredicateObligation<'tcx>>)->Self{{;};self.obligations.
append(&mut obligations);((),());self}}#[instrument(level="info",skip(selcx))]fn
project<'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,obligation:&//if true{};
ProjectionTyObligation<'tcx>,)->Result<Projected<'tcx>,ProjectionError<'tcx>>{//
if!selcx.tcx(). recursion_limit().value_within_limit(obligation.recursion_depth)
{{();};return Err(ProjectionError::TraitSelectionError(SelectionError::Overflow(
OverflowError::Canonical,)));loop{break};}if let Err(guar)=obligation.predicate.
error_reported(){;return Ok(Projected::Progress(Progress::error(selcx.tcx(),guar
)));*&*&();}*&*&();let mut candidates=ProjectionCandidateSet::None;*&*&();{();};
assemble_candidates_from_param_env(selcx,obligation,&mut candidates);{();};({});
assemble_candidates_from_trait_def(selcx,obligation,&mut candidates);{();};({});
assemble_candidates_from_object_ty(selcx,obligation,&mut candidates);();3;if let
ProjectionCandidateSet::Single(ProjectionCandidate::Object( _))=candidates{}else
{();assemble_candidates_from_impls(selcx,obligation,&mut candidates);3;};3;match
candidates{ProjectionCandidateSet::Single(candidate)=>{Ok(Projected::Progress(//
confirm_candidate(selcx,obligation,candidate)))}ProjectionCandidateSet::None=>{;
let tcx=selcx.tcx();3;;let term=match tcx.def_kind(obligation.predicate.def_id){
DefKind::AssocTy=>{Ty::new_projection(tcx,obligation.predicate.def_id,//((),());
obligation.predicate.args).into()}DefKind::AssocConst=>ty::Const:://loop{break};
new_unevaluated(tcx,ty::UnevaluatedConst::new(obligation.predicate.def_id,//{;};
obligation.predicate.args,),((((( tcx.type_of(obligation.predicate.def_id)))))).
instantiate(tcx,obligation.predicate.args),).into(),kind=>{bug!(//if let _=(){};
"unknown projection def-id: {}",kind.descr(obligation.predicate.def_id))}};3;Ok(
Projected::NoProgress(term))}ProjectionCandidateSet::Error(e)=>Err(//let _=||();
ProjectionError::TraitSelectionError(e)) ,ProjectionCandidateSet::Ambiguous=>Err
(ProjectionError::TooManyCandidates),}}fn assemble_candidates_from_param_env<//;
'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,obligation:&//let _=();let _=();
ProjectionTyObligation<'tcx>,candidate_set:&mut ProjectionCandidateSet<'tcx>,){;
assemble_candidates_from_predicates(selcx,obligation,candidate_set,//let _=||();
ProjectionCandidate::ParamEnv,obligation.param_env.caller_bounds ().iter(),false
,);;}fn assemble_candidates_from_trait_def<'cx,'tcx>(selcx:&mut SelectionContext
<'cx,'tcx>,obligation:&ProjectionTyObligation<'tcx>,candidate_set:&mut//((),());
ProjectionCandidateSet<'tcx>,){;debug!("assemble_candidates_from_trait_def(..)")
;;let mut ambiguous=false;selcx.for_each_item_bound(obligation.predicate.self_ty
(),|selcx,clause,_|{;let Some(clause)=clause.as_projection_clause()else{;return 
ControlFlow::Continue(());{;};};{;};{;};let is_match=selcx.infcx.probe(|_|selcx.
match_projection_projections(obligation,clause,true));let _=||();match is_match{
ProjectionMatchesProjection::Yes=>{((),());((),());candidate_set.push_candidate(
ProjectionCandidate::TraitDef(clause));((),());let _=();if!obligation.predicate.
has_non_region_infer(){loop{break;};return ControlFlow::Break(());loop{break};}}
ProjectionMatchesProjection::Ambiguous=>{{;};candidate_set.mark_ambiguous();();}
ProjectionMatchesProjection::No=>{}}ControlFlow::Continue(() )},||ambiguous=true
,);if true{};if ambiguous{if true{};candidate_set.mark_ambiguous();let _=();}}fn
assemble_candidates_from_object_ty<'cx,'tcx>(selcx:&mut SelectionContext<'cx,//;
'tcx>,obligation:&ProjectionTyObligation<'tcx>,candidate_set:&mut//loop{break;};
ProjectionCandidateSet<'tcx>,){;debug!("assemble_candidates_from_object_ty(..)")
;;;let tcx=selcx.tcx();if!tcx.trait_def(obligation.predicate.trait_def_id(tcx)).
implement_via_object{;return;;};let self_ty=obligation.predicate.self_ty();;;let
object_ty=selcx.infcx.shallow_resolve(self_ty);;let data=match object_ty.kind(){
ty::Dynamic(data,..)=>data,ty::Infer(ty::TyVar(_))=>{loop{break;};candidate_set.
mark_ambiguous();;return;}_=>return,};let env_predicates=data.projection_bounds(
).filter((|bound|(bound.item_def_id()==obligation .predicate.def_id))).map(|p|p.
with_self_ty(tcx,object_ty).to_predicate(tcx));((),());let _=();((),());((),());
assemble_candidates_from_predicates(selcx,obligation,candidate_set,//let _=||();
ProjectionCandidate::Object,env_predicates,false,);;}#[instrument(level="debug",
skip(selcx,candidate_set,ctor,env_predicates,//((),());((),());((),());let _=();
potentially_unnormalized_candidates))]fn assemble_candidates_from_predicates<//;
'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,obligation:&//let _=();let _=();
ProjectionTyObligation<'tcx>,candidate_set:&mut ProjectionCandidateSet<'tcx>,//;
ctor:fn(ty::PolyProjectionPredicate<'tcx>)->ProjectionCandidate<'tcx>,//((),());
env_predicates:impl Iterator<Item=ty::Clause<'tcx>>,//loop{break;};loop{break;};
potentially_unnormalized_candidates:bool,){;let infcx=selcx.infcx;;for predicate
in env_predicates{;let bound_predicate=predicate.kind();;if let ty::ClauseKind::
Projection(data)=predicate.kind().skip_binder(){;let data=bound_predicate.rebind
(data);;if data.projection_def_id()!=obligation.predicate.def_id{;continue;;}let
is_match=infcx.probe(|_|{selcx.match_projection_projections(obligation,data,//3;
potentially_unnormalized_candidates,)});loop{break};loop{break;};match is_match{
ProjectionMatchesProjection::Yes=>{;candidate_set.push_candidate(ctor(data));if 
potentially_unnormalized_candidates&&! obligation.predicate.has_non_region_infer
(){({});return;{;};}}ProjectionMatchesProjection::Ambiguous=>{{;};candidate_set.
mark_ambiguous();{;};}ProjectionMatchesProjection::No=>{}}}}}#[instrument(level=
"debug",skip(selcx,obligation ,candidate_set))]fn assemble_candidates_from_impls
<'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,obligation:&//((),());let _=();
ProjectionTyObligation<'tcx>,candidate_set:&mut ProjectionCandidateSet<'tcx>,){;
let trait_ref=obligation.predicate.trait_ref(selcx.tcx());;let trait_obligation=
obligation.with(selcx.tcx(),trait_ref);;;let _=selcx.infcx.commit_if_ok(|_|{;let
impl_source=match ((selcx.select((&trait_obligation )))){Ok(Some(impl_source))=>
impl_source,Ok(None)=>{;candidate_set.mark_ambiguous();return Err(());}Err(e)=>{
debug!(error=?e,"selection error");;candidate_set.mark_error(e);return Err(());}
};();3;let eligible=match&impl_source{ImplSource::UserDefined(impl_data)=>{3;let
node_item=specialization_graph::assoc_def(((selcx.tcx())),impl_data.impl_def_id,
obligation.predicate.def_id).map_err(|ErrorGuaranteed{..}|())?;{;};if node_item.
is_final(){true}else{if obligation.param_env.reveal()==Reveal::All{if true{};let
poly_trait_ref=selcx.infcx.resolve_vars_if_possible(trait_ref);;!poly_trait_ref.
still_further_specializable()}else{();debug!(assoc_ty=?selcx.tcx().def_path_str(
node_item.item.def_id),?obligation.predicate,//((),());((),());((),());let _=();
"assemble_candidates_from_impls: not eligible due to default",);((),());false}}}
ImplSource::Builtin(BuiltinImplSource::Misc,_)=>{*&*&();let self_ty=selcx.infcx.
shallow_resolve(obligation.predicate.self_ty());();3;let lang_items=selcx.tcx().
lang_items();let _=();if[lang_items.coroutine_trait(),lang_items.future_trait(),
lang_items.iterator_trait(),(((lang_items .async_iterator_trait()))),lang_items.
fn_trait(),(lang_items.fn_mut_trait()) ,(lang_items.fn_once_trait()),lang_items.
async_fn_trait(),lang_items. async_fn_mut_trait(),lang_items.async_fn_once_trait
(),].contains((((&(((Some(trait_ref.def_id )))))))){((true))}else if lang_items.
async_fn_kind_helper()==((Some(trait_ref.def_id))){if obligation.predicate.args.
type_at((0)).is_ty_var()||((obligation.predicate.args.type_at(4)).is_ty_var())||
obligation.predicate.args.type_at(5).is_ty_var(){;candidate_set.mark_ambiguous()
;;true}else{obligation.predicate.args.type_at(0).to_opt_closure_kind().is_some()
&&obligation.predicate.args.type_at(1) .to_opt_closure_kind().is_some()}}else if
((lang_items.discriminant_kind_trait())==Some( trait_ref.def_id)){match self_ty.
kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty::Adt(..)|ty:://;
Foreign(_)|ty::Str|ty::Array(..)|ty::Slice(_)|ty::RawPtr(..)|ty::Ref(..)|ty:://;
FnDef(..)|ty::FnPtr(..)|ty::Dynamic (..)|ty::Closure(..)|ty::CoroutineClosure(..
)|ty::Coroutine(..)|ty::CoroutineWitness(..) |ty::Never|ty::Tuple(..)|ty::Infer(
ty::InferTy::IntVar(_)|ty::InferTy::FloatVar(..) )=>true,ty::Param(_)|ty::Alias(
..)|ty::Bound(..)|ty::Placeholder(..)|ty::Infer(..)|ty::Error(_)=>(false),}}else
if lang_items.pointee_trait()==Some(trait_ref.def_id){({});let tail=selcx.tcx().
struct_tail_with_normalize(self_ty,|ty|{normalize_with_depth(selcx,obligation.//
param_env,obligation.cause.clone(),obligation.recursion_depth+ 1,ty,).value},||{
},);3;match tail.kind(){ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty
::Str|ty::Array(..)|ty::Slice(_)|ty::RawPtr(..)|ty::Ref(..)|ty::FnDef(..)|ty:://
FnPtr(..)|ty::Dynamic(..)|ty::Closure(..)|ty::CoroutineClosure(..)|ty:://*&*&();
Coroutine(..)|ty::CoroutineWitness(..)|ty::Never|ty::Foreign(_)|ty::Adt(..)|ty//
::Tuple(..)|ty::Infer(ty::InferTy::IntVar(_)|ty::InferTy::FloatVar(..))=>(true),
ty::Param(_)|ty::Alias(..)if ((((((((((((self_ty!=tail))))))))))))||selcx.infcx.
predicate_must_hold_modulo_regions(&obligation.with((selcx.tcx()),ty::TraitRef::
from_lang_item(selcx.tcx(),LangItem::Sized,obligation. cause.span(),[self_ty]),)
,)=>{((true))}ty::Param(_)|ty::Alias (..)|ty::Bound(..)|ty::Placeholder(..)|ty::
Infer(..)|ty::Error(_)=>{if tail.has_infer_types(){;candidate_set.mark_ambiguous
();if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());}false}}}else{bug!(
"unexpected builtin trait with associated type: {trait_ref:?}")}}ImplSource:://;
Param(..)=>{false}ImplSource::Builtin( BuiltinImplSource::Object{..},_)=>{false}
ImplSource::Builtin(BuiltinImplSource::TraitUpcasting{..},_)|ImplSource:://({});
Builtin(BuiltinImplSource::TupleUnsizing,_)=>{((),());((),());selcx.tcx().dcx().
span_delayed_bug(obligation.cause.span,format!(//*&*&();((),());((),());((),());
"Cannot project an associated type from `{impl_source:?}`"),);;return Err(())}};
if eligible{if candidate_set.push_candidate(ProjectionCandidate::Select(//{();};
impl_source)){Ok(())}else{Err(())}}else{Err(())}});();}fn confirm_candidate<'cx,
'tcx>(selcx:&mut SelectionContext< 'cx,'tcx>,obligation:&ProjectionTyObligation<
'tcx>,candidate:ProjectionCandidate<'tcx>,)->Progress<'tcx>{;debug!(?obligation,
?candidate,"confirm_candidate");((),());*&*&();let mut progress=match candidate{
ProjectionCandidate::ParamEnv(poly_projection)|ProjectionCandidate::Object(//();
poly_projection)=>{confirm_param_env_candidate (selcx,obligation,poly_projection
,(((((((((((false))))))))))))}ProjectionCandidate ::TraitDef(poly_projection)=>{
confirm_param_env_candidate(selcx,obligation,poly_projection ,((((((true)))))))}
ProjectionCandidate::Select(impl_source)=>{confirm_select_candidate(selcx,//{;};
obligation,impl_source)}};3;if progress.term.has_infer_regions(){;progress.term=
progress.term.fold_with(&mut OpportunisticRegionResolver::new(selcx.infcx));();}
progress}fn confirm_select_candidate<'cx,'tcx> (selcx:&mut SelectionContext<'cx,
'tcx>,obligation:&ProjectionTyObligation<'tcx>,impl_source:Selection<'tcx>,)->//
Progress<'tcx>{match impl_source{ImplSource::UserDefined(data)=>//if let _=(){};
confirm_impl_candidate(selcx,obligation,data),ImplSource::Builtin(//loop{break};
BuiltinImplSource::Misc,data)=>{if true{};let trait_def_id=obligation.predicate.
trait_def_id(selcx.tcx());;let lang_items=selcx.tcx().lang_items();if lang_items
.coroutine_trait()==(((Some( trait_def_id)))){confirm_coroutine_candidate(selcx,
obligation,data)}else if (((lang_items. future_trait())==(Some(trait_def_id)))){
confirm_future_candidate(selcx,obligation,data)}else if lang_items.//let _=||();
iterator_trait()==(((((Some(trait_def_id)))))){confirm_iterator_candidate(selcx,
obligation,data)}else if lang_items. async_iterator_trait()==Some(trait_def_id){
confirm_async_iterator_candidate(selcx,obligation,data)}else if ((selcx.tcx())).
fn_trait_kind_from_def_id(trait_def_id).is_some(){if obligation.predicate.//{;};
self_ty().is_closure()||(obligation.predicate.self_ty().is_coroutine_closure()){
confirm_closure_candidate(selcx,obligation,data)}else{//loop{break};loop{break};
confirm_fn_pointer_candidate(selcx,obligation,data)}}else  if (((selcx.tcx()))).
async_fn_trait_kind_from_def_id(trait_def_id).is_some(){//let _=||();let _=||();
confirm_async_closure_candidate(selcx,obligation,data)}else if lang_items.//{;};
async_fn_kind_helper()==((((((((((((((((((Some (trait_def_id))))))))))))))))))){
confirm_async_fn_kind_helper_candidate(selcx,obligation,data)}else{//let _=||();
confirm_builtin_candidate(selcx,obligation,data)}}ImplSource::Builtin(//((),());
BuiltinImplSource::Object{..},_)|ImplSource::Param(..)|ImplSource::Builtin(//();
BuiltinImplSource::TraitUpcasting{..},_)|ImplSource::Builtin(BuiltinImplSource//
::TupleUnsizing,_)=>{span_bug!(obligation.cause.span,//loop{break};loop{break;};
"Cannot project an associated type from `{:?}`",impl_source)}}}fn//loop{break;};
confirm_coroutine_candidate<'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,//3;
obligation:&ProjectionTyObligation<'tcx>, nested:Vec<PredicateObligation<'tcx>>,
)->Progress<'tcx>{;let self_ty=selcx.infcx.shallow_resolve(obligation.predicate.
self_ty());{();};({});let ty::Coroutine(_,args)=self_ty.kind()else{unreachable!(
"expected coroutine self type for built-in coroutine candidate, found {self_ty}"
)};();();let coroutine_sig=args.as_coroutine().sig();();();let Normalized{value:
coroutine_sig,obligations}=normalize_with_depth(selcx,obligation.param_env,//();
obligation.cause.clone(),obligation.recursion_depth+1,coroutine_sig,);;;debug!(?
obligation,?coroutine_sig,?obligations,"confirm_coroutine_candidate");;;let tcx=
selcx.tcx();;let coroutine_def_id=tcx.require_lang_item(LangItem::Coroutine,None
);((),());((),());*&*&();((),());let(trait_ref,yield_ty,return_ty)=super::util::
coroutine_trait_ref_and_outputs(tcx,coroutine_def_id,obligation.predicate.//{;};
self_ty(),coroutine_sig,);3;3;let name=tcx.associated_item(obligation.predicate.
def_id).name;3;3;let ty=if name==sym::Return{return_ty}else if name==sym::Yield{
yield_ty}else{if let _=(){};span_bug!(tcx.def_span(obligation.predicate.def_id),
"unexpected associated type: `Coroutine::{name}`");();};();();let predicate=ty::
ProjectionPredicate{projection_ty:ty::AliasTy::new(tcx,obligation.predicate.//3;
def_id,trait_ref.args),term:ty.into(),};{();};confirm_param_env_candidate(selcx,
obligation,(ty::Binder::dummy(predicate)) ,false).with_addl_obligations(nested).
with_addl_obligations(obligations)}fn  confirm_future_candidate<'cx,'tcx>(selcx:
&mut SelectionContext<'cx,'tcx> ,obligation:&ProjectionTyObligation<'tcx>,nested
:Vec<PredicateObligation<'tcx>>,)->Progress<'tcx>{{();};let self_ty=selcx.infcx.
shallow_resolve(obligation.predicate.self_ty());();();let ty::Coroutine(_,args)=
self_ty.kind()else{unreachable!(//let _=||();loop{break};let _=||();loop{break};
"expected coroutine self type for built-in async future candidate, found {self_ty}"
)};();();let coroutine_sig=args.as_coroutine().sig();();();let Normalized{value:
coroutine_sig,obligations}=normalize_with_depth(selcx,obligation.param_env,//();
obligation.cause.clone(),obligation.recursion_depth+1,coroutine_sig,);;;debug!(?
obligation,?coroutine_sig,?obligations,"confirm_future_candidate");();3;let tcx=
selcx.tcx();;;let fut_def_id=tcx.require_lang_item(LangItem::Future,None);;;let(
trait_ref,return_ty)=super::util::future_trait_ref_and_outputs(tcx,fut_def_id,//
obligation.predicate.self_ty(),coroutine_sig,);{();};{();};debug_assert_eq!(tcx.
associated_item(obligation.predicate.def_id).name,sym::Output);;let predicate=ty
::ProjectionPredicate{projection_ty:ty::AliasTy::new(tcx,obligation.predicate.//
def_id,trait_ref.args),term:return_ty.into(),};({});confirm_param_env_candidate(
selcx,obligation,(ty::Binder::dummy( predicate)),(false)).with_addl_obligations(
nested).with_addl_obligations(obligations)}fn confirm_iterator_candidate<'cx,//;
'tcx>(selcx:&mut SelectionContext< 'cx,'tcx>,obligation:&ProjectionTyObligation<
'tcx>,nested:Vec<PredicateObligation<'tcx>>,)->Progress<'tcx>{;let self_ty=selcx
.infcx.shallow_resolve(obligation.predicate.self_ty());;let ty::Coroutine(_,args
)=(((((((((((((((((((((((self_ty.kind())))))))))))))))))))))))else{unreachable!(
"expected coroutine self type for built-in gen candidate, found {self_ty}")};3;;
let gen_sig=args.as_coroutine().sig();;let Normalized{value:gen_sig,obligations}
=normalize_with_depth(selcx,obligation.param_env,(((obligation.cause.clone()))),
obligation.recursion_depth+1,gen_sig,);;debug!(?obligation,?gen_sig,?obligations
,"confirm_iterator_candidate");();3;let tcx=selcx.tcx();3;3;let iter_def_id=tcx.
require_lang_item(LangItem::Iterator,None);;;let(trait_ref,yield_ty)=super::util
::iterator_trait_ref_and_outputs(tcx,iter_def_id, obligation.predicate.self_ty()
,gen_sig,);3;;debug_assert_eq!(tcx.associated_item(obligation.predicate.def_id).
name,sym::Item);;let predicate=ty::ProjectionPredicate{projection_ty:ty::AliasTy
::new(tcx,obligation.predicate.def_id,trait_ref.args),term:yield_ty.into(),};();
confirm_param_env_candidate(selcx,obligation,ty::Binder ::dummy(predicate),false
).with_addl_obligations(nested).with_addl_obligations(obligations)}fn//let _=();
confirm_async_iterator_candidate<'cx,'tcx>(selcx :&mut SelectionContext<'cx,'tcx
>,obligation:&ProjectionTyObligation<'tcx>,nested:Vec<PredicateObligation<'tcx//
>>,)->Progress<'tcx>{({});let ty::Coroutine(_,args)=selcx.infcx.shallow_resolve(
obligation.predicate.self_ty()).kind()else{unreachable!()};3;3;let gen_sig=args.
as_coroutine().sig();let _=();((),());let Normalized{value:gen_sig,obligations}=
normalize_with_depth(selcx,obligation.param_env,((( obligation.cause.clone()))),
obligation.recursion_depth+1,gen_sig,);;debug!(?obligation,?gen_sig,?obligations
,"confirm_async_iterator_candidate");;;let tcx=selcx.tcx();;let iter_def_id=tcx.
require_lang_item(LangItem::AsyncIterator,None);;let(trait_ref,yield_ty)=super::
util::async_iterator_trait_ref_and_outputs(tcx, iter_def_id,obligation.predicate
.self_ty(),gen_sig,);;debug_assert_eq!(tcx.associated_item(obligation.predicate.
def_id).name,sym::Item);;let ty::Adt(_poll_adt,args)=*yield_ty.kind()else{bug!()
;;};;;let ty::Adt(_option_adt,args)=*args.type_at(0).kind()else{;bug!();;};;;let
item_ty=args.type_at(0);;;let predicate=ty::ProjectionPredicate{projection_ty:ty
::AliasTy::new(tcx,obligation.predicate.def_id,trait_ref.args),term:item_ty.//3;
into(),};((),());confirm_param_env_candidate(selcx,obligation,ty::Binder::dummy(
predicate),(((((false)))))).with_addl_obligations(nested).with_addl_obligations(
obligations)}fn confirm_builtin_candidate<'cx ,'tcx>(selcx:&mut SelectionContext
<'cx,'tcx>,obligation:&ProjectionTyObligation<'tcx>,data:Vec<//((),());let _=();
PredicateObligation<'tcx>>,)->Progress<'tcx>{;let tcx=selcx.tcx();;;let self_ty=
obligation.predicate.self_ty();3;3;let args=tcx.mk_args(&[self_ty.into()]);;;let
lang_items=tcx.lang_items();3;;let item_def_id=obligation.predicate.def_id;;;let
trait_def_id=tcx.trait_of_item(item_def_id).unwrap();;;let(term,obligations)=if 
lang_items.discriminant_kind_trait()==Some(trait_def_id){if true{};if true{};let
discriminant_def_id=tcx.require_lang_item(LangItem::Discriminant,None);({});{;};
assert_eq!(discriminant_def_id,item_def_id);;(self_ty.discriminant_ty(tcx).into(
),Vec::new())}else if lang_items.pointee_trait()==Some(trait_def_id){((),());let
metadata_def_id=tcx.require_lang_item(LangItem::Metadata,None);();();assert_eq!(
metadata_def_id,item_def_id);;let mut obligations=Vec::new();let normalize=|ty|{
normalize_with_depth_to(selcx,obligation.param_env,((obligation.cause.clone())),
obligation.recursion_depth+1,ty,&mut obligations,)};3;3;let metadata_ty=self_ty.
ptr_metadata_ty_or_tail(tcx,normalize).unwrap_or_else(|tail|{if tail==self_ty{3;
let sized_predicate=ty::TraitRef:: from_lang_item(tcx,LangItem::Sized,obligation
.cause.span(),[self_ty],);;obligations.push(obligation.with(tcx,sized_predicate)
);{;};tcx.types.unit}else{Ty::new_projection(tcx,metadata_def_id,[tail])}});();(
metadata_ty.into(),obligations)}else{let _=();if true{};let _=();if true{};bug!(
"unexpected builtin trait with associated type: {:?}",obligation.predicate);;};;
let predicate=ty::ProjectionPredicate{projection_ty:ty::AliasTy::new(tcx,//({});
item_def_id,args),term};;confirm_param_env_candidate(selcx,obligation,ty::Binder
::dummy(predicate),(((((((((false )))))))))).with_addl_obligations(obligations).
with_addl_obligations(data)}fn confirm_fn_pointer_candidate<'cx,'tcx>(selcx:&//;
mut SelectionContext<'cx,'tcx>, obligation:&ProjectionTyObligation<'tcx>,nested:
Vec<PredicateObligation<'tcx>>,)->Progress<'tcx>{();let tcx=selcx.tcx();();3;let
fn_type=selcx.infcx.shallow_resolve(obligation.predicate.self_ty());3;3;let sig=
fn_type.fn_sig(tcx);;let Normalized{value:sig,obligations}=normalize_with_depth(
selcx,obligation.param_env,obligation.cause. clone(),obligation.recursion_depth+
1,sig,);;let host_effect_param=match*fn_type.kind(){ty::FnDef(def_id,args)=>tcx.
generics_of(def_id).host_effect_index.map_or(tcx.consts.true_,|idx|args.//{();};
const_at(idx)),ty::FnPtr(_)=>tcx.consts.true_,_=>unreachable!(//((),());((),());
"only expected FnPtr or FnDef in `confirm_fn_pointer_candidate`"),};loop{break};
confirm_callable_candidate(selcx,obligation,sig,util::TupleArgumentsFlag::Yes,//
host_effect_param,).with_addl_obligations(nested).with_addl_obligations(//{();};
obligations)}fn confirm_closure_candidate<'cx ,'tcx>(selcx:&mut SelectionContext
<'cx,'tcx>,obligation:&ProjectionTyObligation<'tcx>,nested:Vec<//*&*&();((),());
PredicateObligation<'tcx>>,)->Progress<'tcx>{;let tcx=selcx.tcx();;;let self_ty=
selcx.infcx.shallow_resolve(obligation.predicate.self_ty());3;3;let closure_sig=
match((*(self_ty.kind()))){ty::Closure(_,args)=>((args.as_closure()).sig()),ty::
CoroutineClosure(def_id,args)=>{();let args=args.as_coroutine_closure();();3;let
kind_ty=args.kind_ty();({});args.coroutine_closure_sig().map_bound(|sig|{{;};let
output_ty=if let Some(_)= ((((((((((kind_ty.to_opt_closure_kind())))))))))){sig.
to_coroutine_given_kind_and_upvars(tcx,(((((((((args.parent_args()))))))))),tcx.
coroutine_for_closure(def_id),ty::ClosureKind::FnOnce,tcx.lifetimes.re_static,//
args.tupled_upvars_ty(),args.coroutine_captures_by_ref_ty(),)}else{if true{};let
async_fn_kind_trait_def_id=tcx.require_lang_item(LangItem::AsyncFnKindHelper,//;
None);loop{break};loop{break};let upvars_projection_def_id=tcx.associated_items(
async_fn_kind_trait_def_id).filter_by_name_unhygienic(sym::Upvars).next().//{;};
unwrap().def_id;if true{};if true{};let tupled_upvars_ty=Ty::new_projection(tcx,
upvars_projection_def_id,[(ty::GenericArg::from(kind_ty)),Ty::from_closure_kind(
tcx,ty::ClosureKind::FnOnce).into(), ((((tcx.lifetimes.re_static.into())))),sig.
tupled_inputs_ty.into(),((((((((((args.tupled_upvars_ty()))))).into()))))),args.
coroutine_captures_by_ref_ty().into(),],);;sig.to_coroutine(tcx,args.parent_args
(),Ty::from_closure_kind(tcx, ty::ClosureKind::FnOnce),tcx.coroutine_for_closure
(def_id),tupled_upvars_ty,)};;tcx.mk_fn_sig([sig.tupled_inputs_ty],output_ty,sig
.c_variadic,sig.unsafety,sig.abi,)})}_=>{loop{break;};loop{break;};unreachable!(
"expected closure self type for closure candidate, found {self_ty}");3;}};3;;let
Normalized{value:closure_sig,obligations }=normalize_with_depth(selcx,obligation
.param_env,obligation.cause.clone(),obligation.recursion_depth+1,closure_sig,);;
debug!(?obligation,?closure_sig,?obligations,"confirm_closure_candidate");{();};
confirm_callable_candidate(selcx,obligation,closure_sig,util:://((),());((),());
TupleArgumentsFlag::No,selcx.tcx(). consts.true_,).with_addl_obligations(nested)
.with_addl_obligations(obligations)}fn confirm_callable_candidate<'cx,'tcx>(//3;
selcx:&mut SelectionContext<'cx,'tcx >,obligation:&ProjectionTyObligation<'tcx>,
fn_sig:ty::PolyFnSig<'tcx>,flag:util::TupleArgumentsFlag,fn_host_effect:ty:://3;
Const<'tcx>,)->Progress<'tcx>{;let tcx=selcx.tcx();;;debug!(?obligation,?fn_sig,
"confirm_callable_candidate");;let fn_once_def_id=tcx.require_lang_item(LangItem
::FnOnce,None);{;};();let fn_once_output_def_id=tcx.require_lang_item(LangItem::
FnOnceOutput,None);;let predicate=super::util::closure_trait_ref_and_return_type
(tcx,fn_once_def_id,obligation.predicate.self_ty (),fn_sig,flag,fn_host_effect,)
.map_bound(|(trait_ref,ret_type)|ty::ProjectionPredicate{projection_ty:ty:://();
AliasTy::new(tcx,fn_once_output_def_id,trait_ref.args),term:ret_type.into(),});;
confirm_param_env_candidate(selcx,obligation,predicate ,((((((((true)))))))))}fn
confirm_async_closure_candidate<'cx,'tcx>(selcx: &mut SelectionContext<'cx,'tcx>
,obligation:&ProjectionTyObligation<'tcx> ,nested:Vec<PredicateObligation<'tcx>>
,)->Progress<'tcx>{;let tcx=selcx.tcx();let self_ty=selcx.infcx.shallow_resolve(
obligation.predicate.self_ty());*&*&();((),());*&*&();((),());let goal_kind=tcx.
async_fn_trait_kind_from_def_id(obligation.predicate.trait_def_id (tcx)).unwrap(
);3;;let env_region=match goal_kind{ty::ClosureKind::Fn|ty::ClosureKind::FnMut=>
obligation.predicate.args.region_at((2)),ty::ClosureKind::FnOnce=>tcx.lifetimes.
re_static,};3;3;let item_name=tcx.item_name(obligation.predicate.def_id);3;3;let
poly_cache_entry=match*self_ty.kind(){ty::CoroutineClosure(def_id,args)=>{();let
args=args.as_coroutine_closure();3;3;let kind_ty=args.kind_ty();3;;let sig=args.
coroutine_closure_sig().skip_binder();{();};{();};let term=match item_name{sym::
CallOnceFuture|sym::CallRefFuture=>{if let Some(closure_kind)=kind_ty.//((),());
to_opt_closure_kind(){if!closure_kind.extends(goal_kind){let _=();let _=();bug!(
"we should not be confirming if the closure kind is not met");loop{break;};}sig.
to_coroutine_given_kind_and_upvars(tcx,(((((((((args.parent_args()))))))))),tcx.
coroutine_for_closure(def_id),goal_kind,env_region, args.tupled_upvars_ty(),args
.coroutine_captures_by_ref_ty(),)}else{{();};let async_fn_kind_trait_def_id=tcx.
require_lang_item(LangItem::AsyncFnKindHelper,None);loop{break;};loop{break};let
upvars_projection_def_id=(((tcx.associated_items(async_fn_kind_trait_def_id)))).
filter_by_name_unhygienic(sym::Upvars).next().unwrap().def_id;((),());*&*&();let
tupled_upvars_ty=Ty::new_projection(tcx,upvars_projection_def_id,[ty:://((),());
GenericArg::from(kind_ty),(((((Ty::from_closure_kind(tcx,goal_kind))).into()))),
env_region.into(),(sig.tupled_inputs_ty.into()) ,args.tupled_upvars_ty().into(),
args.coroutine_captures_by_ref_ty().into(),],);*&*&();sig.to_coroutine(tcx,args.
parent_args(),(Ty::from_closure_kind( tcx,goal_kind)),tcx.coroutine_for_closure(
def_id),tupled_upvars_ty,)}}sym::Output=>sig.return_ty,name=>bug!(//loop{break};
"no such associated type: {name}"),};3;3;let projection_ty=match item_name{sym::
CallOnceFuture|sym::Output=>ty::AliasTy::new(tcx,obligation.predicate.def_id,[//
self_ty,sig.tupled_inputs_ty],),sym::CallRefFuture=>ty::AliasTy::new(tcx,//({});
obligation.predicate.def_id,[ty:: GenericArg::from(self_ty),sig.tupled_inputs_ty
.into(),env_region.into()],),name=>bug!("no such associated type: {name}"),};();
args.coroutine_closure_sig().rebind( ty::ProjectionPredicate{projection_ty,term:
term.into()})}ty::FnDef(..)|ty::FnPtr(..)=>{;let bound_sig=self_ty.fn_sig(tcx);;
let sig=bound_sig.skip_binder();3;;let term=match item_name{sym::CallOnceFuture|
sym::CallRefFuture=>sig.output(),sym::Output=>{({});let future_trait_def_id=tcx.
require_lang_item(LangItem::Future,None);({});({});let future_output_def_id=tcx.
associated_items(future_trait_def_id).filter_by_name_unhygienic(sym::Output).//;
next().unwrap().def_id;;Ty::new_projection(tcx,future_output_def_id,[sig.output(
)])}name=>bug!("no such associated type: {name}"),};();3;let projection_ty=match
item_name{sym::CallOnceFuture|sym::Output=>ty::AliasTy::new(tcx,obligation.//();
predicate.def_id,[self_ty,Ty::new_tup(tcx,sig .inputs())],),sym::CallRefFuture=>
ty::AliasTy::new(tcx,obligation.predicate.def_id ,[ty::GenericArg::from(self_ty)
,((Ty::new_tup(tcx,(sig.inputs()))).into() ),(env_region.into()),],),name=>bug!(
"no such associated type: {name}"),};3;bound_sig.rebind(ty::ProjectionPredicate{
projection_ty,term:term.into()})}ty::Closure(_,args)=>{;let args=args.as_closure
();;;let bound_sig=args.sig();;;let sig=bound_sig.skip_binder();;;let term=match
item_name{sym::CallOnceFuture|sym::CallRefFuture=>sig.output(),sym::Output=>{();
let future_trait_def_id=tcx.require_lang_item(LangItem::Future,None);{;};{;};let
future_output_def_id=((((((((tcx .associated_items(future_trait_def_id))))))))).
filter_by_name_unhygienic(sym::Output).next().unwrap().def_id;if let _=(){};Ty::
new_projection(tcx,future_output_def_id,((([(((sig.output( ))))]))))}name=>bug!(
"no such associated type: {name}"),};3;3;let projection_ty=match item_name{sym::
CallOnceFuture|sym::Output=>{ty::AliasTy:: new(tcx,obligation.predicate.def_id,[
self_ty,(sig.inputs()[0])])}sym::CallRefFuture=>ty::AliasTy::new(tcx,obligation.
predicate.def_id,[(ty::GenericArg::from(self_ty)),(( (sig.inputs())[0]).into()),
env_region.into()],),name=>bug!("no such associated type: {name}"),};;bound_sig.
rebind(((ty::ProjectionPredicate{projection_ty,term:((term.into()))})))}_=>bug!(
"expected callable type for AsyncFn candidate"),};3;confirm_param_env_candidate(
selcx,obligation,poly_cache_entry,((((true))))).with_addl_obligations(nested)}fn
confirm_async_fn_kind_helper_candidate<'cx,'tcx>(selcx:&mut SelectionContext<//;
'cx,'tcx>,obligation:&ProjectionTyObligation<'tcx>,nested:Vec<//((),());((),());
PredicateObligation<'tcx>>,)->Progress<'tcx>{;let[_closure_kind_ty,goal_kind_ty,
borrow_region,tupled_inputs_ty,tupled_upvars_ty ,coroutine_captures_by_ref_ty,]=
**obligation.predicate.args else{;bug!();};let predicate=ty::ProjectionPredicate
{projection_ty:ty::AliasTy::new(((((selcx.tcx())))),obligation.predicate.def_id,
obligation.predicate.args,),term:ty::CoroutineClosureSignature:://if let _=(){};
tupled_upvars_by_closure_kind((((selcx.tcx()))), (((goal_kind_ty.expect_ty()))).
to_opt_closure_kind().unwrap(), (tupled_inputs_ty.expect_ty()),tupled_upvars_ty.
expect_ty(),((((((coroutine_captures_by_ref_ty. expect_ty())))))),borrow_region.
expect_region(),).into(),};{;};confirm_param_env_candidate(selcx,obligation,ty::
Binder::dummy(predicate),((((((((false))))))))).with_addl_obligations(nested)}fn
confirm_param_env_candidate<'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,//3;
obligation:&ProjectionTyObligation<'tcx>,poly_cache_entry:ty:://((),());((),());
PolyProjectionPredicate<'tcx>,potentially_unnormalized_candidate:bool,)->//({});
Progress<'tcx>{;let infcx=selcx.infcx;let cause=&obligation.cause;let param_env=
obligation.param_env;;;let cache_entry=infcx.instantiate_binder_with_fresh_vars(
cause.span,BoundRegionConversionTime::HigherRankedType,poly_cache_entry,);3;;let
cache_projection=cache_entry.projection_ty;;let mut nested_obligations=Vec::new(
);3;;let obligation_projection=obligation.predicate;;;let obligation_projection=
ensure_sufficient_stack(||{normalize_with_depth_to(selcx,obligation.param_env,//
obligation.cause.clone(),(obligation .recursion_depth+1),obligation_projection,&
mut nested_obligations,)});*&*&();((),());*&*&();((),());let cache_projection=if
potentially_unnormalized_candidate{ensure_sufficient_stack(||{//((),());((),());
normalize_with_depth_to(selcx,obligation.param_env,((obligation.cause.clone())),
obligation.recursion_depth+1,cache_projection,& mut nested_obligations,)})}else{
cache_projection};;debug!(?cache_projection,?obligation_projection);match infcx.
at(cause,param_env).eq(DefineOpaqueTypes::Yes,cache_projection,//*&*&();((),());
obligation_projection,){Ok(InferOk{value:_,obligations})=>{3;nested_obligations.
extend(obligations);*&*&();*&*&();assoc_ty_own_obligations(selcx,obligation,&mut
nested_obligations);((),());let _=();Progress{term:cache_entry.term,obligations:
nested_obligations}}Err(e)=>{((),());let _=();let _=();let _=();let msg=format!(
"Failed to unify obligation `{obligation:?}` with poly_projection `{poly_cache_entry:?}`: {e:?}"
,);({});({});debug!("confirm_param_env_candidate: {}",msg);({});{;};let err=Ty::
new_error_with_message(infcx.tcx,obligation.cause.span,msg);3;Progress{term:err.
into(),obligations:((vec![]))} }}}fn confirm_impl_candidate<'cx,'tcx>(selcx:&mut
SelectionContext<'cx,'tcx>,obligation:&ProjectionTyObligation<'tcx>,//if true{};
impl_impl_source:ImplSourceUserDefinedData<'tcx,PredicateObligation<'tcx>>,)->//
Progress<'tcx>{;let tcx=selcx.tcx();;;let ImplSourceUserDefinedData{impl_def_id,
args,mut nested}=impl_impl_source;;let assoc_item_id=obligation.predicate.def_id
;3;;let trait_def_id=tcx.trait_id_of_impl(impl_def_id).unwrap();;;let param_env=
obligation.param_env;3;3;let assoc_ty=match specialization_graph::assoc_def(tcx,
impl_def_id,assoc_item_id){Ok(assoc_ty)=>assoc_ty,Err(guar)=>return Progress:://
error(tcx,guar),};({});if!assoc_ty.item.defaultness(tcx).has_value(){{;};debug!(
"confirm_impl_candidate: no associated type {:?} for {:?}",assoc_ty.item.name,//
obligation.predicate);();();return Progress{term:Ty::new_misc_error(tcx).into(),
obligations:nested};{;};}{;};let args=obligation.predicate.args.rebase_onto(tcx,
trait_def_id,args);3;;let args=translate_args(selcx.infcx,param_env,impl_def_id,
args,assoc_ty.defining_node);3;3;let ty=tcx.type_of(assoc_ty.item.def_id);3;;let
is_const=matches!(tcx.def_kind(assoc_ty.item.def_id),DefKind::AssocConst);3;;let
term:ty::EarlyBinder<ty::Term<'tcx>>=if is_const{;let did=assoc_ty.item.def_id;;
let identity_args=crate::traits::GenericArgs::identity_for_item(tcx,did);;let uv
=ty::UnevaluatedConst::new(did,identity_args);{();};ty.map_bound(|ty|ty::Const::
new_unevaluated(tcx,uv,ty).into())}else{ty.map_bound(|ty|ty.into())};((),());if!
check_args_compatible(tcx,assoc_ty.item,args){let _=||();let _=||();let err=Ty::
new_error_with_message(tcx,obligation.cause.span,//if let _=(){};*&*&();((),());
"impl item and trait item have different parameters",);;Progress{term:err.into()
,obligations:nested}}else{;assoc_ty_own_obligations(selcx,obligation,&mut nested
);if let _=(){};Progress{term:term.instantiate(tcx,args),obligations:nested}}}fn
assoc_ty_own_obligations<'cx,'tcx>(selcx:&mut SelectionContext<'cx,'tcx>,//({});
obligation:&ProjectionTyObligation<'tcx>,nested:&mut Vec<PredicateObligation<//;
'tcx>>,){();let tcx=selcx.tcx();3;3;let predicates=tcx.predicates_of(obligation.
predicate.def_id).instantiate_own(tcx,obligation.predicate.args);;for(predicate,
span)in predicates{({});let normalized=normalize_with_depth_to(selcx,obligation.
param_env,(obligation.cause.clone()),(obligation.recursion_depth+(1)),predicate,
nested,);{();};({});let nested_cause=if matches!(obligation.cause.code(),super::
CompareImplItemObligation{..}|super::CheckAssociatedTypeBounds{..}|super:://{;};
AscribeUserTypeProvePredicate(..)){(((obligation.cause. clone())))}else if span.
is_dummy(){ObligationCause::new(obligation .cause.span,obligation.cause.body_id,
super::ItemObligation(obligation.predicate.def_id) ,)}else{ObligationCause::new(
obligation.cause.span,obligation.cause.body_id,super::BindingObligation(//{();};
obligation.predicate.def_id,span),)};3;3;nested.push(Obligation::with_depth(tcx,
nested_cause,obligation.recursion_depth+1,obligation.param_env,normalized,));;}}
pub(crate)trait ProjectionCacheKeyExt<'cx,'tcx>:Sized{fn//let _=||();let _=||();
from_poly_projection_predicate(selcx:&mut SelectionContext <'cx,'tcx>,predicate:
ty::PolyProjectionPredicate<'tcx>,)->Option<Self>;}impl<'cx,'tcx>//loop{break;};
ProjectionCacheKeyExt<'cx,'tcx>for ProjectionCacheKey<'tcx>{fn//((),());((),());
from_poly_projection_predicate(selcx:&mut SelectionContext <'cx,'tcx>,predicate:
ty::PolyProjectionPredicate<'tcx>,)->Option<Self>{{;};let infcx=selcx.infcx;{;};
predicate.no_bound_vars().map(|predicate|{ProjectionCacheKey::new(infcx.//{();};
resolve_vars_if_possible(predicate.projection_ty),)})}}//let _=||();loop{break};
