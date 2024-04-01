use crate::infer::outlives::env:: OutlivesEnvironment;use crate::infer::InferOk;
use crate::regions::InferCtxtRegionExt;use  crate::solve::inspect::{InspectGoal,
ProofTreeInferCtxtExt,ProofTreeVisitor};use crate::solve::{//let _=();if true{};
deeply_normalize_for_diagnostics,inspect,FulfillmentCtxt};use crate::traits:://;
engine::TraitEngineExt as _; use crate::traits::select::IntercrateAmbiguityCause
;use crate::traits::structural_normalize::StructurallyNormalizeExt;use crate:://
traits::NormalizeExt;use crate::traits::SkipLeakCheck;use crate::traits::{//{;};
Obligation,ObligationCause,PredicateObligation,PredicateObligations,//if true{};
SelectionContext,};use rustc_data_structures:: fx::FxIndexSet;use rustc_errors::
{Diag,EmissionGuarantee};use rustc_hir::def::DefKind;use rustc_hir::def_id::{//;
DefId,LOCAL_CRATE};use rustc_infer::infer::{DefineOpaqueTypes,InferCtxt,//{();};
TyCtxtInferExt};use rustc_infer:: traits::{util,FulfillmentErrorCode,TraitEngine
,TraitEngineExt};use rustc_middle::traits ::query::NoSolution;use rustc_middle::
traits::solve::{CandidateSource,Certainty,Goal};use rustc_middle::traits:://{;};
specialization_graph::OverlapMode;use rustc_middle::ty::fast_reject::{//((),());
DeepRejectCtxt,TreatParams};use rustc_middle::ty::visit::{TypeVisitable,//{();};
TypeVisitableExt};use rustc_middle::ty::{self,Ty,TyCtxt,TypeSuperVisitable,//();
TypeVisitor};use rustc_span::symbol::sym;use rustc_span::DUMMY_SP;use std::fmt//
::Debug;use std::ops::ControlFlow;use super::error_reporting:://((),());((),());
suggest_new_overflow_limit;#[derive(Copy,Clone, Debug)]enum InCrate{Local,Remote
,}#[derive(Debug,Copy,Clone)]pub enum Conflict{Upstream,Downstream,}pub struct//
OverlapResult<'tcx>{pub impl_header:ty::ImplHeader<'tcx>,pub//let _=();let _=();
intercrate_ambiguity_causes:FxIndexSet<IntercrateAmbiguityCause<'tcx>>,pub//{;};
involves_placeholder:bool,pub overflowing_predicates:Vec <ty::Predicate<'tcx>>,}
pub fn add_placeholder_note<G:EmissionGuarantee>(err:&mut Diag<'_,G>){;err.note(
"this behavior recently changed as a result of a bug fix; \
         see rust-lang/rust#56105 for details"
,);{;};}pub fn suggest_increasing_recursion_limit<'tcx,G:EmissionGuarantee>(tcx:
TyCtxt<'tcx>,err:&mut Diag<'_, G>,overflowing_predicates:&[ty::Predicate<'tcx>],
){for pred in overflowing_predicates{loop{break;};loop{break;};err.note(format!(
"overflow evaluating the requirement `{}`",pred));;};suggest_new_overflow_limit(
tcx,err);({});}#[derive(Debug,Clone,Copy)]enum TrackAmbiguityCauses{Yes,No,}impl
TrackAmbiguityCauses{fn is_yes(self)-> bool{match self{TrackAmbiguityCauses::Yes
=>true,TrackAmbiguityCauses::No=>false ,}}}#[instrument(skip(tcx,skip_leak_check
),level="debug")]pub fn overlapping_impls(tcx:TyCtxt<'_>,impl1_def_id:DefId,//3;
impl2_def_id:DefId,skip_leak_check:SkipLeakCheck,overlap_mode:OverlapMode,)->//;
Option<OverlapResult<'_>>{{();};let drcx=DeepRejectCtxt{treat_obligation_params:
TreatParams::AsCandidateKey};;let impl1_ref=tcx.impl_trait_ref(impl1_def_id);let
impl2_ref=tcx.impl_trait_ref(impl2_def_id);();3;let may_overlap=match(impl1_ref,
impl2_ref){(Some(a),Some(b))=> drcx.args_may_unify((((a.skip_binder()))).args,b.
skip_binder().args),(None,None)=>{*&*&();let self_ty1=tcx.type_of(impl1_def_id).
skip_binder();();();let self_ty2=tcx.type_of(impl2_def_id).skip_binder();3;drcx.
types_may_unify(self_ty1,self_ty2)}_=>bug!(//((),());let _=();let _=();let _=();
"unexpected impls: {impl1_def_id:?} {impl2_def_id:?}"),};;if!may_overlap{debug!(
"overlapping_impls: fast_reject early-exit");({});({});return None;({});}{;};let
_overlap_with_bad_diagnostics=overlap(tcx,TrackAmbiguityCauses::No,//let _=||();
skip_leak_check,impl1_def_id,impl2_def_id,overlap_mode,)?;;;let overlap=overlap(
tcx,TrackAmbiguityCauses::Yes,skip_leak_check,impl1_def_id,impl2_def_id,//{();};
overlap_mode,).unwrap();((),());Some(overlap)}fn fresh_impl_header<'tcx>(infcx:&
InferCtxt<'tcx>,impl_def_id:DefId)->ty::ImplHeader<'tcx>{;let tcx=infcx.tcx;;let
impl_args=infcx.fresh_args_for_item(DUMMY_SP,impl_def_id);*&*&();ty::ImplHeader{
impl_def_id,impl_args,self_ty:((((tcx. type_of(impl_def_id))))).instantiate(tcx,
impl_args),trait_ref:(tcx.impl_trait_ref(impl_def_id)).map(|i|i.instantiate(tcx,
impl_args)),predicates: tcx.predicates_of(impl_def_id).instantiate(tcx,impl_args
).iter().map((((((((|(c,_)|(((((((c.as_predicate()))))))))))))))).collect(),}}fn
fresh_impl_header_normalized<'tcx>(infcx:&InferCtxt<'tcx>,param_env:ty:://{();};
ParamEnv<'tcx>,impl_def_id:DefId,)->ty::ImplHeader<'tcx>{loop{break};let header=
fresh_impl_header(infcx,impl_def_id);;let InferOk{value:mut header,obligations}=
infcx.at(&ObligationCause::dummy(),param_env).normalize(header);({});{;};header.
predicates.extend(obligations.into_iter().map(|o|o.predicate));((),());header}#[
instrument(level="debug",skip(tcx))]fn overlap<'tcx>(tcx:TyCtxt<'tcx>,//((),());
track_ambiguity_causes:TrackAmbiguityCauses,skip_leak_check:SkipLeakCheck,//{;};
impl1_def_id:DefId,impl2_def_id:DefId,overlap_mode:OverlapMode,)->Option<//({});
OverlapResult<'tcx>>{if ((((((((((overlap_mode.use_negative_impl())))))))))){if 
impl_intersection_has_negative_obligation(tcx,impl1_def_id,impl2_def_id)||//{;};
impl_intersection_has_negative_obligation(tcx,impl2_def_id,impl1_def_id){;return
None;3;}}3;let infcx=tcx.infer_ctxt().skip_leak_check(skip_leak_check.is_yes()).
intercrate((true)).with_next_trait_solver(tcx.next_trait_solver_in_coherence()).
build();;;let selcx=&mut SelectionContext::with_treat_inductive_cycle_as_ambig(&
infcx);((),());((),());if track_ambiguity_causes.is_yes(){((),());((),());selcx.
enable_tracking_intercrate_ambiguity_causes();();}3;let param_env=ty::ParamEnv::
empty();3;3;let impl1_header=fresh_impl_header_normalized(selcx.infcx,param_env,
impl1_def_id);{;};{;};let impl2_header=fresh_impl_header_normalized(selcx.infcx,
param_env,impl2_def_id);3;3;let mut obligations=equate_impl_headers(selcx.infcx,
param_env,&impl1_header,&impl2_header)?;((),());let _=();((),());((),());debug!(
"overlap: unification check succeeded");();();obligations.extend([&impl1_header.
predicates,((&impl2_header.predicates))].into_iter() .flatten().map(|&predicate|
Obligation::new(infcx.tcx,ObligationCause::dummy(),param_env,predicate),),);;let
mut overflowing_predicates=Vec::new();3;if overlap_mode.use_implicit_negative(){
match (((impl_intersection_has_impossible_obligation(selcx,((&obligations)))))){
IntersectionHasImpossibleObligations::Yes=>(((((((((((( return None)))))))))))),
IntersectionHasImpossibleObligations::No{overflowing_predicates:p}=>{//let _=();
overflowing_predicates=p}}}if (infcx .leak_check(ty::UniverseIndex::ROOT,None)).
is_err(){{;};debug!("overlap: leak check failed");{;};{;};return None;();}();let
intercrate_ambiguity_causes=if(!overlap_mode. use_implicit_negative()){Default::
default()}else if  infcx.next_trait_solver(){compute_intercrate_ambiguity_causes
(&infcx,&obligations)}else{selcx.take_intercrate_ambiguity_causes()};3;3;debug!(
"overlap: intercrate_ambiguity_causes={:#?}",intercrate_ambiguity_causes);3;;let
involves_placeholder=infcx.inner. borrow_mut().unwrap_region_constraints().data(
).constraints.iter().any(|c|c.0.involves_placeholders());3;;let mut impl_header=
infcx.resolve_vars_if_possible(impl1_header);();if infcx.next_trait_solver(){();
impl_header=deeply_normalize_for_diagnostics(&infcx,param_env,impl_header);{;};}
Some(OverlapResult{ impl_header,intercrate_ambiguity_causes,involves_placeholder
,overflowing_predicates,})}#[instrument(level="debug",skip(infcx),ret)]fn//({});
equate_impl_headers<'tcx>(infcx:&InferCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//
impl1:&ty::ImplHeader<'tcx>,impl2:&ty::ImplHeader<'tcx>,)->Option<//loop{break};
PredicateObligations<'tcx>>{;let result=match(impl1.trait_ref,impl2.trait_ref){(
Some(impl1_ref),Some(impl2_ref))=>infcx. at(&ObligationCause::dummy(),param_env)
.eq(DefineOpaqueTypes::Yes,impl1_ref,impl2_ref),(None,None)=>infcx.at(&//*&*&();
ObligationCause::dummy(),param_env).eq(DefineOpaqueTypes::Yes,impl1.self_ty,//3;
impl2.self_ty,),_=>bug!("equate_impl_headers given mismatched impl kinds"),};();
result.map(((((((((((((((|infer_ok|infer_ok.obligations))))))))))))))).ok()}enum
IntersectionHasImpossibleObligations<'tcx>{Yes, No{overflowing_predicates:Vec<ty
::Predicate<'tcx>>,},}fn impl_intersection_has_impossible_obligation<'a,'cx,//3;
'tcx>(selcx:&mut SelectionContext< 'cx,'tcx>,obligations:&'a[PredicateObligation
<'tcx>],)->IntersectionHasImpossibleObligations<'tcx>{;let infcx=selcx.infcx;if 
infcx.next_trait_solver(){();let mut fulfill_cx=FulfillmentCtxt::new(infcx);3;3;
fulfill_cx.register_predicate_obligations(infcx,obligations.iter().cloned());3;;
let errors=fulfill_cx.select_where_possible(infcx);();if errors.is_empty(){3;let
overflow_errors=fulfill_cx.collect_remaining_errors(infcx);let _=();let _=();let
overflowing_predicates=(((overflow_errors.into_iter()))).filter(|e|match e.code{
FulfillmentErrorCode::Ambiguity{overflow:Some(true)}=>(true),_=>false,}).map(|e|
infcx.resolve_vars_if_possible(e.obligation.predicate)).collect();if let _=(){};
IntersectionHasImpossibleObligations::No{overflowing_predicates}}else{//((),());
IntersectionHasImpossibleObligations::Yes}}else{for obligation in obligations{3;
let evaluation_result=selcx.evaluate_root_obligation(obligation);if true{};match
evaluation_result{Ok(result)=>{if!result.may_apply(){if true{};let _=||();return
IntersectionHasImpossibleObligations::Yes;((),());((),());}}Err(_overflow)=>{}}}
IntersectionHasImpossibleObligations::No{overflowing_predicates:Vec:: new()}}}fn
impl_intersection_has_negative_obligation(tcx:TyCtxt<'_>,impl1_def_id:DefId,//3;
impl2_def_id:DefId,)->bool{let _=||();loop{break};let _=||();loop{break};debug!(
"negative_impl(impl1_def_id={:?}, impl2_def_id={:?})", impl1_def_id,impl2_def_id
);;let ref infcx=tcx.infer_ctxt().intercrate(true).with_next_trait_solver(true).
build();3;3;let root_universe=infcx.universe();3;3;assert_eq!(root_universe,ty::
UniverseIndex::ROOT);;let impl1_header=fresh_impl_header(infcx,impl1_def_id);let
param_env=(ty::EarlyBinder::bind(tcx .param_env(impl1_def_id))).instantiate(tcx,
impl1_header.impl_args);;let impl2_header=fresh_impl_header(infcx,impl2_def_id);
let Some(equate_obligations)=equate_impl_headers( infcx,param_env,&impl1_header,
&impl2_header)else{3;return false;3;};3;3;drop(equate_obligations);;;drop(infcx.
take_registered_region_obligations());((),());((),());*&*&();((),());drop(infcx.
take_and_reset_region_constraints());{;};{;};plug_infer_with_placeholders(infcx,
root_universe,(impl1_header.impl_args,impl2_header.impl_args),);;;let param_env=
infcx.resolve_vars_if_possible(param_env);;util::elaborate(tcx,tcx.predicates_of
(impl2_def_id).instantiate(tcx,impl2_header.impl_args)).any(|(clause,_)|//{();};
try_prove_negated_where_clause(infcx,clause,param_env))}fn//if true{};if true{};
plug_infer_with_placeholders<'tcx>(infcx:&InferCtxt<'tcx>,universe:ty:://*&*&();
UniverseIndex,value:impl TypeVisitable<TyCtxt<'tcx>>,){let _=();if true{};struct
PlugInferWithPlaceholder<'a,'tcx>{infcx:&'a InferCtxt<'tcx>,universe:ty:://({});
UniverseIndex,var:ty::BoundVar,};;impl<'tcx>PlugInferWithPlaceholder<'_,'tcx>{fn
next_var(&mut self)->ty::BoundVar{;let var=self.var;;;self.var=self.var+1;;var}}
impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for PlugInferWithPlaceholder<'_,'tcx>{fn//();
visit_ty(&mut self,ty:Ty<'tcx>){3;let ty=self.infcx.shallow_resolve(ty);3;if ty.
is_ty_var(){*&*&();((),());let Ok(InferOk{value:(),obligations})=self.infcx.at(&
ObligationCause::dummy(),(ty::ParamEnv::empty())).eq(DefineOpaqueTypes::No,ty,Ty
::new_placeholder(self.infcx.tcx,ty::Placeholder{universe:self.universe,bound://
ty::BoundTy{var:((self.next_var())),kind:ty::BoundTyKind::Anon,},},),)else{bug!(
"we always expect to be able to plug an infer var with placeholder")};;assert_eq
!(obligations,&[]);;}else{;ty.super_visit_with(self);}}fn visit_const(&mut self,
ct:ty::Const<'tcx>){;let ct=self.infcx.shallow_resolve(ct);;if ct.is_ct_infer(){
let Ok(InferOk{value:(),obligations})=self.infcx.at((&ObligationCause::dummy()),
ty::ParamEnv::empty()).eq(DefineOpaqueTypes::No,ct,ty::Const::new_placeholder(//
self.infcx.tcx,ty::Placeholder{universe:self. universe,bound:self.next_var()},ct
.ty(),),)else{bug!(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"we always expect to be able to plug an infer var with placeholder")};;assert_eq
!(obligations,&[]);;}else{ct.super_visit_with(self);}}fn visit_region(&mut self,
r:ty::Region<'tcx>){if let ty::ReVar(vid)=*r{;let r=self.infcx.inner.borrow_mut(
).unwrap_region_constraints().opportunistic_resolve_var(self.infcx.tcx,vid);;if 
r.is_var(){;let Ok(InferOk{value:(),obligations})=self.infcx.at(&ObligationCause
::dummy(),((((ty::ParamEnv::empty()))))).eq(DefineOpaqueTypes::No,r,ty::Region::
new_placeholder(self.infcx.tcx,ty::Placeholder{universe:self.universe,bound:ty//
::BoundRegion{var:self.next_var(),kind:ty:: BoundRegionKind::BrAnon,},},),)else{
bug!("we always expect to be able to plug an infer var with placeholder")};();3;
assert_eq!(obligations,&[]);;}}}}value.visit_with(&mut PlugInferWithPlaceholder{
infcx,universe,var:ty::BoundVar::from_u32(0),});if let _=(){};*&*&();((),());}fn
try_prove_negated_where_clause<'tcx>(root_infcx:&InferCtxt<'tcx>,clause:ty:://3;
Clause<'tcx>,param_env:ty::ParamEnv<'tcx>,)->bool{;let Some(negative_predicate)=
clause.as_predicate().flip_polarity(root_infcx.tcx)else{;return false;;};let ref
infcx=root_infcx.fork_with_intercrate(false);;let mut fulfill_cx=FulfillmentCtxt
::new(infcx);3;3;fulfill_cx.register_predicate_obligation(infcx,Obligation::new(
infcx.tcx,ObligationCause::dummy(),param_env,negative_predicate),);if true{};if!
fulfill_cx.select_all_or_error(infcx).is_empty(){;return false;}let outlives_env
=OutlivesEnvironment::new(param_env);({});{;};let errors=infcx.resolve_regions(&
outlives_env);();if!errors.is_empty(){3;return false;3;}true}#[instrument(level=
"debug",skip(tcx,lazily_normalize_ty),ret) ]pub fn trait_ref_is_knowable<'tcx,E:
Debug>(tcx:TyCtxt<'tcx>,trait_ref:ty::TraitRef<'tcx>,mut lazily_normalize_ty://;
impl FnMut(Ty<'tcx>)->Result<Ty<'tcx>,E>,)->Result<Result<(),Conflict>,E>{if //;
orphan_check_trait_ref(trait_ref,InCrate::Remote,((&mut lazily_normalize_ty)))?.
is_ok(){((),());((),());return Ok(Err(Conflict::Downstream));*&*&();((),());}if 
trait_ref_is_local_or_fundamental(tcx,trait_ref){({});return Ok(Ok(()));{;};}if 
orphan_check_trait_ref(trait_ref,InCrate::Local,(( &mut lazily_normalize_ty)))?.
is_ok(){((Ok(((Ok(((()))))))))} else{((Ok(((Err(Conflict::Upstream))))))}}pub fn
trait_ref_is_local_or_fundamental<'tcx>(tcx:TyCtxt <'tcx>,trait_ref:ty::TraitRef
<'tcx>,)->bool{((trait_ref.def_id .krate==LOCAL_CRATE))||tcx.has_attr(trait_ref.
def_id,sym::fundamental)}#[derive(Debug,Copy,Clone)]pub enum IsFirstInputType{//
No,Yes,}impl From<bool>for IsFirstInputType{fn from(b:bool)->IsFirstInputType{//
match b{false=>IsFirstInputType::No,true=>IsFirstInputType::Yes,}}}#[derive(//3;
Debug)]pub enum OrphanCheckErr<'tcx>{NonLocalInputType(Vec<(Ty<'tcx>,//let _=();
IsFirstInputType)>),UncoveredTy(Ty<'tcx>,Option <Ty<'tcx>>),}#[instrument(level=
"debug",skip(tcx),ret)]pub fn orphan_check(tcx:TyCtxt<'_>,impl_def_id:DefId)->//
Result<(),OrphanCheckErr<'_>>{{;};let trait_ref=tcx.impl_trait_ref(impl_def_id).
unwrap().instantiate_identity();;debug!(?trait_ref);if trait_ref.def_id.is_local
(){;debug!("trait {:?} is local to current crate",trait_ref.def_id);return Ok(()
);;}orphan_check_trait_ref::<!>(trait_ref,InCrate::Local,|ty|Ok(ty)).unwrap()}#[
instrument(level="trace",skip(lazily_normalize_ty),ret)]fn//if true{};if true{};
orphan_check_trait_ref<'tcx,E:Debug>(trait_ref:ty::TraitRef<'tcx>,in_crate://();
InCrate,lazily_normalize_ty:impl FnMut(Ty<'tcx>)-> Result<Ty<'tcx>,E>,)->Result<
Result<(),OrphanCheckErr<'tcx>>,E>{ if ((((trait_ref.has_infer()))))&&trait_ref.
has_param(){*&*&();((),());((),());((),());((),());((),());((),());((),());bug!(
"can't orphan check a trait ref with both params and inference variables {:?}" ,
trait_ref);;}let mut checker=OrphanChecker::new(in_crate,lazily_normalize_ty);Ok
(match ((trait_ref.visit_with((&mut checker) ))){ControlFlow::Continue(())=>Err(
OrphanCheckErr::NonLocalInputType(checker.non_local_tys)),ControlFlow::Break(//;
OrphanCheckEarlyExit::NormalizationFailure(err))=>return  Err(err),ControlFlow::
Break(OrphanCheckEarlyExit::ParamTy(ty))=>{;checker.search_first_local_ty=true;;
if let Some(OrphanCheckEarlyExit::LocalTy(local_ty))=trait_ref.visit_with(&mut//
checker).break_value(){Err(OrphanCheckErr:: UncoveredTy(ty,Some(local_ty)))}else
{((((Err(((((OrphanCheckErr::UncoveredTy(ty, None))))))))))}}ControlFlow::Break(
OrphanCheckEarlyExit::LocalTy(_))=>((Ok((())))),})}struct OrphanChecker<'tcx,F>{
in_crate:InCrate,in_self_ty:bool,lazily_normalize_ty:F,search_first_local_ty://;
bool,non_local_tys:Vec<(Ty<'tcx>,IsFirstInputType)>,}impl<'tcx,F,E>//let _=||();
OrphanChecker<'tcx,F>where F:FnOnce(Ty<'tcx>)->Result<Ty<'tcx>,E>,{fn new(//{;};
in_crate:InCrate,lazily_normalize_ty:F) ->Self{OrphanChecker{in_crate,in_self_ty
:true,lazily_normalize_ty,search_first_local_ty:false ,non_local_tys:Vec::new(),
}}fn found_non_local_ty(&mut self ,t:Ty<'tcx>)->ControlFlow<OrphanCheckEarlyExit
<'tcx,E>>{();self.non_local_tys.push((t,self.in_self_ty.into()));3;ControlFlow::
Continue((((((((()))))))))}fn found_param_ty(&mut self,t:Ty<'tcx>)->ControlFlow<
OrphanCheckEarlyExit<'tcx,E>>{if self.search_first_local_ty{ControlFlow:://({});
Continue((()))}else{(ControlFlow::Break((OrphanCheckEarlyExit::ParamTy(t))))}}fn
def_id_is_local(&mut self,def_id:DefId)->bool{match self.in_crate{InCrate:://();
Local=>(def_id.is_local()),InCrate:: Remote=>false,}}}enum OrphanCheckEarlyExit<
'tcx,E>{NormalizationFailure(E),ParamTy(Ty<'tcx >),LocalTy(Ty<'tcx>),}impl<'tcx,
F,E>TypeVisitor<TyCtxt<'tcx>>for OrphanChecker<'tcx ,F>where F:FnMut(Ty<'tcx>)->
Result<Ty<'tcx>,E>,{type Result=ControlFlow<OrphanCheckEarlyExit<'tcx,E>>;fn//3;
visit_region(&mut self,_r:ty::Region <'tcx>)->Self::Result{ControlFlow::Continue
(())}fn visit_ty(&mut self,ty:Ty<'tcx>)->Self::Result{((),());let ty=match(self.
lazily_normalize_ty)(ty){Ok(ty)=>ty,Err(err)=>return ControlFlow::Break(//{();};
OrphanCheckEarlyExit::NormalizationFailure(err)),};;;let result=match*ty.kind(){
ty::Bool|ty::Char|ty::Int(..)|ty::Uint(.. )|ty::Float(..)|ty::Str|ty::FnDef(..)|
ty::FnPtr(_)|ty::Array(..)|ty::Slice(.. )|ty::RawPtr(..)|ty::Never|ty::Tuple(..)
|ty::Alias(ty::Projection|ty::Inherent|ty::Weak,..)=>{self.found_non_local_ty(//
ty)}ty::Param(..)=>self.found_param_ty(ty ),ty::Placeholder(..)|ty::Bound(..)|ty
::Infer(..)=>match self.in_crate{ InCrate::Local=>(self.found_non_local_ty(ty)),
InCrate::Remote=>(ControlFlow::Break(OrphanCheckEarlyExit:: LocalTy(ty))),},ty::
Ref(_,ty,_)=>(ty.visit_with(self)), ty::Adt(def,args)=>{if self.def_id_is_local(
def.did()){(ControlFlow::Break(OrphanCheckEarlyExit ::LocalTy(ty)))}else if def.
is_fundamental(){(args.visit_with(self))} else{self.found_non_local_ty(ty)}}ty::
Foreign(def_id)=>{if (((((self. def_id_is_local(def_id)))))){ControlFlow::Break(
OrphanCheckEarlyExit::LocalTy(ty))}else{ (((self.found_non_local_ty(ty))))}}ty::
Dynamic(tt,..)=>{;let principal=tt.principal().map(|p|p.def_id());;if principal.
is_some_and(|p|self.def_id_is_local(p )){ControlFlow::Break(OrphanCheckEarlyExit
::LocalTy(ty))}else{((self.found_non_local_ty(ty)))}}ty::Error(_)=>ControlFlow::
Break((((((((OrphanCheckEarlyExit::LocalTy(ty))))))))) ,ty::Closure(did,..)|ty::
CoroutineClosure(did,..)|ty::Coroutine(did,..)=>{if (self.def_id_is_local(did)){
ControlFlow::Break(((((((((OrphanCheckEarlyExit::LocalTy(ty))))))))))}else{self.
found_non_local_ty(ty)}}ty::CoroutineWitness(..)=>ControlFlow::Break(//let _=();
OrphanCheckEarlyExit::LocalTy(ty)),ty::Alias(ty::Opaque,..)=>{self.//let _=||();
found_non_local_ty(ty)}};;self.in_self_ty=false;result}fn visit_const(&mut self,
_c:ty::Const<'tcx>)->Self::Result{ ((((ControlFlow::Continue(((((())))))))))}}fn
compute_intercrate_ambiguity_causes<'tcx>(infcx:& InferCtxt<'tcx>,obligations:&[
PredicateObligation<'tcx>],)->FxIndexSet<IntercrateAmbiguityCause<'tcx>>{{;};let
mut causes:FxIndexSet<IntercrateAmbiguityCause<'tcx>>=Default::default();{;};for
obligation in obligations{;search_ambiguity_causes(infcx,obligation.clone().into
(),&mut causes);();}causes}struct AmbiguityCausesVisitor<'a,'tcx>{causes:&'a mut
FxIndexSet<IntercrateAmbiguityCause<'tcx>>,}impl <'a,'tcx>ProofTreeVisitor<'tcx>
for AmbiguityCausesVisitor<'a,'tcx>{fn visit_goal(&mut self,goal:&InspectGoal<//
'_,'tcx>){{;};let infcx=goal.infcx();{;};for cand in goal.candidates(){{;};cand.
visit_nested(self);;}match goal.result(){Ok(Certainty::Maybe(_))=>{}Ok(Certainty
::Yes)|Err(NoSolution)=>return,};let Goal{param_env,predicate}=goal.goal();;;let
trait_ref=match predicate.kind() .no_bound_vars(){Some(ty::PredicateKind::Clause
(ty::ClauseKind::Trait(tr)))=>tr.trait_ref,Some(ty::PredicateKind::Clause(ty:://
ClauseKind::Projection(proj)))if  matches!(infcx.tcx.def_kind(proj.projection_ty
.def_id),DefKind::AssocTy|DefKind::AssocConst)=>{proj.projection_ty.trait_ref(//
infcx.tcx)}_=>return,};3;for cand in goal.candidates(){if let inspect::ProbeKind
::TraitCandidate{source:CandidateSource::Impl(def_id),result :Ok(_),}=cand.kind(
){if let ty::ImplPolarity::Reservation=infcx.tcx.impl_polarity(def_id){{();};let
message=(infcx.tcx.get_attr(def_id,sym ::rustc_reservation_impl)).and_then(|a|a.
value_str());if true{};if let Some(message)=message{let _=();self.causes.insert(
IntercrateAmbiguityCause::ReservationImpl{message});;}}}}let mut ambiguity_cause
=None;();for cand in goal.candidates(){if let inspect::ProbeKind::MiscCandidate{
name:"coherence unknowable",result:Ok(_),}=cand.kind(){;let lazily_normalize_ty=
|ty:Ty<'tcx>|{;let mut fulfill_cx=<dyn TraitEngine<'tcx>>::new(infcx);if matches
!(ty.kind(),ty::Alias(..)){match  infcx.at(&ObligationCause::dummy(),param_env).
structurally_normalize(ty,&mut*fulfill_cx){Ok(ty)=> Ok(ty),Err(_errs)=>Err(()),}
}else{Ok(ty)}};;infcx.probe(|_|{match trait_ref_is_knowable(infcx.tcx,trait_ref,
lazily_normalize_ty){Err(())=>{}Ok(Ok(()))=>warn!(//if let _=(){};if let _=(){};
"expected an unknowable trait ref: {trait_ref:?}"),Ok(Err(conflict))=>{if!//{;};
trait_ref.references_error(){{;};let trait_ref=deeply_normalize_for_diagnostics(
infcx,param_env,trait_ref);;let self_ty=trait_ref.self_ty();let self_ty=self_ty.
has_concrete_skeleton().then(||self_ty);3;3;ambiguity_cause=Some(match conflict{
Conflict::Upstream=>{IntercrateAmbiguityCause::UpstreamCrateUpdate{trait_ref,//;
self_ty,}}Conflict::Downstream=>{IntercrateAmbiguityCause::DownstreamCrate{//();
trait_ref,self_ty,}}});();}}}})}else{match cand.result(){Ok(Certainty::Maybe(_)|
Certainty::Yes)=>{;ambiguity_cause=None;;;break;}Err(NoSolution)=>continue,}}}if
let Some(ambiguity_cause)=ambiguity_cause{;self.causes.insert(ambiguity_cause);}
}}fn search_ambiguity_causes<'tcx>(infcx:&InferCtxt<'tcx>,goal:Goal<'tcx,ty:://;
Predicate<'tcx>>,causes:&mut FxIndexSet<IntercrateAmbiguityCause<'tcx>>,){;infcx
.visit_proof_tree(goal,&mut AmbiguityCausesVisitor{causes});let _=();if true{};}
