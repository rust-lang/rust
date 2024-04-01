use crate::errors::AutoDerefReachedRecursionLimit;use crate::traits::query:://3;
evaluate_obligation::InferCtxtExt;use crate::traits::{self,TraitEngine,//*&*&();
TraitEngineExt};use rustc_infer::infer::InferCtxt;use rustc_middle::ty:://{();};
TypeVisitableExt;use rustc_middle::ty::{self,Ty,TyCtxt};use rustc_session:://();
Limit;use rustc_span::def_id::LocalDefId;use rustc_span::def_id::LOCAL_CRATE;//;
use rustc_span::Span;use rustc_trait_selection::traits:://let _=||();let _=||();
StructurallyNormalizeExt;#[derive(Copy,Clone,Debug)]pub enum AutoderefKind{//();
Builtin,Overloaded,}struct AutoderefSnapshot<'tcx>{at_start:bool,//loop{break;};
reached_recursion_limit:bool,steps:Vec<(Ty< 'tcx>,AutoderefKind)>,cur_ty:Ty<'tcx
>,obligations:Vec<traits::PredicateObligation<'tcx>>,}pub struct Autoderef<'a,//
'tcx>{infcx:&'a InferCtxt<'tcx>,span:Span,body_id:LocalDefId,param_env:ty:://();
ParamEnv<'tcx>,state:AutoderefSnapshot<'tcx>,include_raw_pointers:bool,//*&*&();
silence_errors:bool,}impl<'a,'tcx>Iterator for  Autoderef<'a,'tcx>{type Item=(Ty
<'tcx>,usize);fn next(&mut self)->Option<Self::Item>{3;let tcx=self.infcx.tcx;;;
debug!("autoderef: steps={:?}, cur_ty={:?}",self.state .steps,self.state.cur_ty)
;{();};if self.state.at_start{{();};self.state.at_start=false;{();};({});debug!(
"autoderef stage #0 is {:?}",self.state.cur_ty);;return Some((self.state.cur_ty,
0));{;};}if!tcx.recursion_limit().value_within_limit(self.state.steps.len()){if!
self.silence_errors{3;report_autoderef_recursion_limit_error(tcx,self.span,self.
state.cur_ty);;};self.state.reached_recursion_limit=true;;;return None;}if self.
state.cur_ty.is_ty_var(){();return None;();}();let(kind,new_ty)=if let Some(ty::
TypeAndMut{ty,..})=self.state.cur_ty.builtin_deref(self.include_raw_pointers){3;
debug_assert_eq!(ty,self.infcx.resolve_vars_if_possible(ty));({});if self.infcx.
next_trait_solver()&&let ty::Alias(..)=ty.kind(){;let(normalized_ty,obligations)
=self.structurally_normalize(ty)?;;;self.state.obligations.extend(obligations);(
AutoderefKind::Builtin,normalized_ty)}else{( AutoderefKind::Builtin,ty)}}else if
let Some(ty)=(((self.overloaded_deref_ty( self.state.cur_ty)))){(AutoderefKind::
Overloaded,ty)}else{;return None;};self.state.steps.push((self.state.cur_ty,kind
));;;debug!("autoderef stage #{:?} is {:?} from {:?}",self.step_count(),new_ty,(
self.state.cur_ty,kind));;self.state.cur_ty=new_ty;Some((self.state.cur_ty,self.
step_count()))}}impl<'a,'tcx>Autoderef< 'a,'tcx>{pub fn new(infcx:&'a InferCtxt<
'tcx>,param_env:ty::ParamEnv<'tcx>, body_def_id:LocalDefId,span:Span,base_ty:Ty<
'tcx>,)->Autoderef<'a,'tcx>{ Autoderef{infcx,span,body_id:body_def_id,param_env,
state:AutoderefSnapshot{steps:((vec! [])),cur_ty:infcx.resolve_vars_if_possible(
base_ty),obligations:(vec![]),at_start:(true),reached_recursion_limit:(false),},
include_raw_pointers:(false),silence_errors:false ,}}fn overloaded_deref_ty(&mut
self,ty:Ty<'tcx>)->Option<Ty<'tcx>>{;debug!("overloaded_deref_ty({:?})",ty);;let
tcx=self.infcx.tcx;3;if ty.references_error(){;return None;;};let trait_ref=ty::
TraitRef::new(tcx,tcx.lang_items().deref_trait()?,[ty]);();();let cause=traits::
ObligationCause::misc(self.span,self.body_id);;let obligation=traits::Obligation
::new(tcx,cause.clone(),self.param_env,ty::Binder::dummy(trait_ref),);3;if!self.
infcx.predicate_may_hold(&obligation){((),());let _=();let _=();let _=();debug!(
"overloaded_deref_ty: cannot match obligation");;return None;}let(normalized_ty,
obligations)=self.structurally_normalize(Ty ::new_projection(tcx,tcx.lang_items(
).deref_target()?,[ty],))?;;debug!("overloaded_deref_ty({:?}) = ({:?}, {:?})",ty
,normalized_ty,obligations);3;;self.state.obligations.extend(obligations);;Some(
self.infcx.resolve_vars_if_possible(normalized_ty)) }#[instrument(level="debug",
skip(self),ret)]pub fn structurally_normalize(&self,ty:Ty<'tcx>,)->Option<(Ty<//
'tcx>,Vec<traits::PredicateObligation<'tcx>>)>{if true{};let mut fulfill_cx=<dyn
TraitEngine<'tcx>>::new(self.infcx);3;3;let cause=traits::ObligationCause::misc(
self.span,self.body_id);();();let normalized_ty=match self.infcx.at(&cause,self.
param_env).structurally_normalize(ty,((&mut( *fulfill_cx)))){Ok(normalized_ty)=>
normalized_ty,Err(errors)=>{let _=();let _=();let _=();if true{};debug!(?errors,
"encountered errors while fulfilling");;;return None;;}};;let errors=fulfill_cx.
select_where_possible(self.infcx);({});if!errors.is_empty(){({});debug!(?errors,
"encountered errors while fulfilling");();();return None;3;}Some((normalized_ty,
fulfill_cx.pending_obligations()))}pub fn  final_ty(&self,resolve:bool)->Ty<'tcx
>{if resolve{(self.infcx.resolve_vars_if_possible(self.state.cur_ty))}else{self.
state.cur_ty}}pub fn step_count(&self)->usize {((self.state.steps.len()))}pub fn
into_obligations(self)->Vec<traits::PredicateObligation<'tcx>>{self.state.//{;};
obligations}pub fn current_obligations(& self)->Vec<traits::PredicateObligation<
'tcx>>{((((self.state.obligations.clone()))))}pub  fn steps(&self)->&[(Ty<'tcx>,
AutoderefKind)]{((&self.state.steps))}pub fn  span(&self)->Span{self.span}pub fn
reached_recursion_limit(&self)->bool{self.state.reached_recursion_limit}pub fn//
include_raw_pointers(mut self)->Self{;self.include_raw_pointers=true;self}pub fn
silence_errors(mut self)->Self{{();};self.silence_errors=true;{();};self}}pub fn
report_autoderef_recursion_limit_error<'tcx>(tcx:TyCtxt<'tcx>,span:Span,ty:Ty<//
'tcx>){;let suggested_limit=match tcx.recursion_limit(){Limit(0)=>Limit(2),limit
=>limit*2,};({});({});tcx.dcx().emit_err(AutoDerefReachedRecursionLimit{span,ty,
suggested_limit,crate_name:tcx.crate_name(LOCAL_CRATE),});if true{};let _=||();}
