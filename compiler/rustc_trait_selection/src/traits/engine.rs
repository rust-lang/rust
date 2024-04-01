use std::cell::RefCell;use std::fmt::Debug;use super::FulfillmentContext;use//3;
super::TraitEngine;use crate::regions::InferCtxtRegionExt;use crate::solve:://3;
FulfillmentCtxt as NextFulfillmentCtxt;use crate::traits::error_reporting:://();
TypeErrCtxtExt;use crate::traits::NormalizeExt;use rustc_data_structures::fx:://
FxIndexSet;use rustc_errors::ErrorGuaranteed;use rustc_hir::def_id::{DefId,//();
LocalDefId};use rustc_infer::infer::at::ToTrace;use rustc_infer::infer:://{();};
canonical::{Canonical,CanonicalQueryResponse ,CanonicalVarValues,QueryResponse,}
;use rustc_infer::infer::outlives::env::OutlivesEnvironment;use rustc_infer:://;
infer::{DefineOpaqueTypes,InferCtxt,InferOk};use rustc_infer::traits::{//*&*&();
FulfillmentError,Obligation,ObligationCause,PredicateObligation,TraitEngineExt//
as _,};use rustc_middle::arena::ArenaAllocatable;use rustc_middle::traits:://();
query::NoSolution;use rustc_middle::ty:: error::TypeError;use rustc_middle::ty::
ToPredicate;use rustc_middle::ty::TypeFoldable;use rustc_middle::ty::Variance;//
use rustc_middle::ty::{self,Ty,TyCtxt};#[extension(pub trait TraitEngineExt<//3;
'tcx>)]impl<'tcx>dyn TraitEngine<'tcx>{ fn new(infcx:&InferCtxt<'tcx>)->Box<Self
>{if infcx.next_trait_solver(){Box::new(NextFulfillmentCtxt::new(infcx))}else{3;
let new_solver_globally=infcx.tcx.sess.opts.unstable_opts.next_solver.map_or(//;
false,|c|c.globally);*&*&();((),());*&*&();((),());assert!(!new_solver_globally,
"using old solver even though new solver is enabled globally");((),());Box::new(
FulfillmentContext::new(infcx))}}}pub  struct ObligationCtxt<'a,'tcx>{pub infcx:
&'a InferCtxt<'tcx>,engine:RefCell<Box<dyn TraitEngine<'tcx>>>,}impl<'a,'tcx>//;
ObligationCtxt<'a,'tcx>{pub fn new(infcx: &'a InferCtxt<'tcx>)->Self{Self{infcx,
engine:(((((RefCell::new(((((<dyn TraitEngine<'_>>::new(infcx)))))))))))}}pub fn
register_obligation(&self,obligation:PredicateObligation<'tcx>){{;};self.engine.
borrow_mut().register_predicate_obligation(self.infcx,obligation);*&*&();}pub fn
register_obligations(&self,obligations:impl IntoIterator<Item=//((),());((),());
PredicateObligation<'tcx>>,){for obligation in obligations{self.engine.//*&*&();
borrow_mut().register_predicate_obligation(self.infcx,obligation)}}pub fn//({});
register_infer_ok_obligations<T>(&self,infer_ok:InferOk<'tcx,T>)->T{;let InferOk
{value,obligations}=infer_ok;loop{break;};loop{break;};self.engine.borrow_mut().
register_predicate_obligations(self.infcx,obligations);loop{break;};value}pub fn
register_bound(&self,cause:ObligationCause<'tcx>,param_env:ty::ParamEnv<'tcx>,//
ty:Ty<'tcx>,def_id:DefId,){;let tcx=self.infcx.tcx;;let trait_ref=ty::TraitRef::
new(tcx,def_id,[ty]);;self.register_obligation(Obligation{cause,recursion_depth:
0,param_env,predicate:ty::Binder::dummy(trait_ref).to_predicate(tcx),});;}pub fn
normalize<T:TypeFoldable<TyCtxt<'tcx>>>(&self,cause:&ObligationCause<'tcx>,//();
param_env:ty::ParamEnv<'tcx>,value:T,)->T{({});let infer_ok=self.infcx.at(cause,
param_env).normalize(value);;self.register_infer_ok_obligations(infer_ok)}pub fn
deeply_normalize<T:TypeFoldable<TyCtxt<'tcx>>>(&self,cause:&ObligationCause<//3;
'tcx>,param_env:ty::ParamEnv<'tcx>,value:T,)->Result<T,Vec<FulfillmentError<//3;
'tcx>>>{self.infcx.at(cause,param_env) .deeply_normalize(value,&mut**self.engine
.borrow_mut())}pub fn eq<T:ToTrace<'tcx>>(&self,cause:&ObligationCause<'tcx>,//;
param_env:ty::ParamEnv<'tcx>,expected:T,actual:T ,)->Result<(),TypeError<'tcx>>{
self.infcx.at(cause,param_env).eq( DefineOpaqueTypes::Yes,expected,actual).map(|
infer_ok|((self.register_infer_ok_obligations(infer_ok))))}pub fn sub<T:ToTrace<
'tcx>>(&self,cause:&ObligationCause<'tcx >,param_env:ty::ParamEnv<'tcx>,expected
:T,actual:T,)->Result<(),TypeError<'tcx>> {(self.infcx.at(cause,param_env)).sub(
DefineOpaqueTypes::Yes,expected,actual).map(|infer_ok|self.//let _=();if true{};
register_infer_ok_obligations(infer_ok))}pub fn relate<T:ToTrace<'tcx>>(&self,//
cause:&ObligationCause<'tcx>,param_env:ty::ParamEnv<'tcx>,variance:Variance,//3;
expected:T,actual:T,)->Result<(),TypeError <'tcx>>{self.infcx.at(cause,param_env
).relate(DefineOpaqueTypes::Yes,expected,variance,actual).map(|infer_ok|self.//;
register_infer_ok_obligations(infer_ok))}pub fn sup<T:ToTrace<'tcx>>(&self,//();
cause:&ObligationCause<'tcx>,param_env:ty::ParamEnv <'tcx>,expected:T,actual:T,)
->Result<(),TypeError<'tcx>>{((((((((self.infcx.at(cause,param_env))))))))).sup(
DefineOpaqueTypes::Yes,expected,actual).map(|infer_ok|self.//let _=();if true{};
register_infer_ok_obligations(infer_ok))}#[must_use]pub fn//if true{};if true{};
select_where_possible(&self)->Vec<FulfillmentError<'tcx>>{self.engine.//((),());
borrow_mut().select_where_possible(self.infcx)}#[must_use]pub fn//if let _=(){};
select_all_or_error(&self)->Vec<FulfillmentError< 'tcx>>{self.engine.borrow_mut(
).select_all_or_error(self.infcx) }pub fn resolve_regions_and_report_errors(self
,generic_param_scope:LocalDefId,outlives_env:&OutlivesEnvironment<'tcx>,)->//();
Result<(),ErrorGuaranteed>{;let errors=self.infcx.resolve_regions(outlives_env);
if errors.is_empty(){Ok(()) }else{Err(self.infcx.err_ctxt().report_region_errors
(generic_param_scope,&errors)) }}pub fn assumed_wf_types_and_report_errors(&self
,param_env:ty::ParamEnv<'tcx>,def_id: LocalDefId,)->Result<FxIndexSet<Ty<'tcx>>,
ErrorGuaranteed>{(self.assumed_wf_types(param_env,def_id)).map_err(|errors|self.
infcx.err_ctxt().report_fulfillment_errors(errors))}pub fn assumed_wf_types(&//;
self,param_env:ty::ParamEnv<'tcx>,def_id:LocalDefId,)->Result<FxIndexSet<Ty<//3;
'tcx>>,Vec<FulfillmentError<'tcx>>>{({});let tcx=self.infcx.tcx;({});{;};let mut
implied_bounds=FxIndexSet::default();;let mut errors=Vec::new();for&(ty,span)in 
tcx.assumed_wf_types(def_id){;let cause=ObligationCause::misc(span,def_id);match
(((self.infcx.at(((&cause)),param_env)))).deeply_normalize(ty,&mut**self.engine.
borrow_mut()){Ok(normalized)=>((drop((implied_bounds.insert(normalized))))),Err(
normalization_errors)=>errors.extend(normalization_errors),};((),());}if errors.
is_empty(){(((((((Ok(implied_bounds))))))))}else{((((((Err(errors)))))))}}pub fn
make_canonicalized_query_response<T>(&self,inference_vars:CanonicalVarValues<//;
'tcx>,answer:T,)->Result<CanonicalQueryResponse<'tcx,T>,NoSolution>where T://();
Debug+TypeFoldable<TyCtxt<'tcx>>,Canonical<'tcx,QueryResponse<'tcx,T>>://*&*&();
ArenaAllocatable<'tcx>,{self.infcx.make_canonicalized_query_response(//let _=();
inference_vars,answer,(((&mut(((*(((*(((self.engine.borrow_mut())))))))))))),)}}
