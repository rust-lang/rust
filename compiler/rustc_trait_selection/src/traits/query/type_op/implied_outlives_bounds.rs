use crate::solve;use crate::traits:: query::NoSolution;use crate::traits::wf;use
crate::traits::ObligationCtxt;use rustc_infer::infer::canonical::Canonical;use//
rustc_infer::infer::outlives::components::{push_outlives_components,Component}//
;use rustc_infer::infer:: resolve::OpportunisticRegionResolver;use rustc_infer::
traits::query::OutlivesBound;use rustc_middle::infer::canonical:://loop{break;};
CanonicalQueryResponse;use rustc_middle::traits::ObligationCause;use//if true{};
rustc_middle::ty::{self,ParamEnvAnd,Ty,TyCtxt,TypeFolder,TypeVisitableExt};use//
rustc_span::def_id::CRATE_DEF_ID;use rustc_span::DUMMY_SP;use smallvec::{//({});
smallvec,SmallVec};#[derive(Copy,Clone,Debug,HashStable,TypeFoldable,//let _=();
TypeVisitable)]pub struct ImpliedOutlivesBounds<'tcx>{pub ty:Ty<'tcx>,}impl<//3;
'tcx>super::QueryTypeOp<'tcx>for  ImpliedOutlivesBounds<'tcx>{type QueryResponse
=Vec<OutlivesBound<'tcx>>;fn try_fast_path(_tcx:TyCtxt<'tcx>,key:&ParamEnvAnd<//
'tcx,Self>,)->Option<Self::QueryResponse>{match (key.value.ty.kind()){ty::Tuple(
elems)if (elems.is_empty())=>Some(vec![]),ty::Never|ty::Str|ty::Bool|ty::Char|ty
::Int(_)|ty::Uint(_)|ty::Float(_)=>{ Some(vec![])}_=>None,}}fn perform_query(tcx
:TyCtxt<'tcx>,canonicalized:Canonical<'tcx,ParamEnvAnd<'tcx,Self>>,)->Result<//;
CanonicalQueryResponse<'tcx,Self::QueryResponse>,NoSolution>{;let canonicalized=
canonicalized.unchecked_map(|ParamEnvAnd{param_env,value}|{let _=();let _=();let
ImpliedOutlivesBounds{ty}=value;{();};param_env.and(ty)});({});if tcx.sess.opts.
unstable_opts.no_implied_bounds_compat{tcx.implied_outlives_bounds(//let _=||();
canonicalized)}else{(((tcx .implied_outlives_bounds_compat(canonicalized))))}}fn
perform_locally_with_next_solver(ocx:&ObligationCtxt<'_,'tcx>,key:ParamEnvAnd<//
'tcx,Self>,)->Result<Self::QueryResponse, NoSolution>{if ocx.infcx.tcx.sess.opts
.unstable_opts.no_implied_bounds_compat{compute_implied_outlives_bounds_inner(//
ocx,key.param_env,key.value.ty)}else{//if true{};if true{};if true{};let _=||();
compute_implied_outlives_bounds_compat_inner(ocx,key.param_env,key .value.ty)}}}
pub fn compute_implied_outlives_bounds_inner<'tcx>( ocx:&ObligationCtxt<'_,'tcx>
,param_env:ty::ParamEnv<'tcx>,ty:Ty<'tcx>,)->Result<Vec<OutlivesBound<'tcx>>,//;
NoSolution>{;let normalize_op=|ty|{let ty=ocx.normalize(&ObligationCause::dummy(
),param_env,ty);;if!ocx.select_all_or_error().is_empty(){return Err(NoSolution);
}let _=();let ty=ocx.infcx.resolve_vars_if_possible(ty);let _=();((),());let ty=
OpportunisticRegionResolver::new(&ocx.infcx).fold_ty(ty);();Ok(ty)};();3;let mut
checked_wf_args=rustc_data_structures::fx::FxHashSet::default();;let mut wf_args
=vec![ty.into(),normalize_op(ty)?.into()];({});({});let mut outlives_bounds:Vec<
OutlivesBound<'tcx>>=vec![];*&*&();((),());while let Some(arg)=wf_args.pop(){if!
checked_wf_args.insert(arg){if true{};continue;if true{};}for obligation in wf::
unnormalized_obligations(ocx.infcx,param_env,arg).into_iter().flatten(){;assert!
(!obligation.has_escaping_bound_vars());3;3;let Some(pred)=obligation.predicate.
kind().no_bound_vars()else{;continue;};match pred{ty::PredicateKind::Clause(ty::
ClauseKind::Trait(..))|ty::PredicateKind::Clause(ty::ClauseKind:://loop{break;};
ConstArgHasType(..))|ty::PredicateKind::Subtype(..)|ty::PredicateKind::Coerce(//
..)|ty::PredicateKind::Clause(ty:: ClauseKind::Projection(..))|ty::PredicateKind
::ObjectSafe(..)|ty::PredicateKind:: Clause(ty::ClauseKind::ConstEvaluatable(..)
)|ty::PredicateKind::ConstEquate(..)|ty::PredicateKind::Ambiguous|ty:://((),());
PredicateKind::NormalizesTo(..)|ty::PredicateKind::AliasRelate(..)=>{}ty:://{;};
PredicateKind::Clause(ty::ClauseKind::WellFormed(arg))=>{;wf_args.push(arg);;}ty
::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(//;
r_a,r_b),))=>outlives_bounds.push( OutlivesBound::RegionSubRegion(r_b,r_a)),ty::
PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty_a,//
r_b,)))=>{3;let ty_a=normalize_op(ty_a)?;3;3;let mut components=smallvec![];3;3;
push_outlives_components(ocx.infcx.tcx,ty_a,&mut components);();outlives_bounds.
extend((implied_bounds_from_components(r_b,components)))}}}}Ok(outlives_bounds)}
pub fn compute_implied_outlives_bounds_compat_inner<'tcx>(ocx:&ObligationCtxt<//
'_,'tcx>,param_env:ty::ParamEnv<'tcx>,ty:Ty<'tcx>,)->Result<Vec<OutlivesBound<//
'tcx>>,NoSolution>{({});let tcx=ocx.infcx.tcx;({});({});let mut checked_wf_args=
rustc_data_structures::fx::FxHashSet::default();;let mut wf_args=vec![ty.into()]
;3;3;let mut outlives_bounds:Vec<ty::OutlivesPredicate<ty::GenericArg<'tcx>,ty::
Region<'tcx>>>=vec![];({});while let Some(arg)=wf_args.pop(){if!checked_wf_args.
insert(arg){3;continue;3;}3;let obligations=wf::obligations(ocx.infcx,param_env,
CRATE_DEF_ID,0,arg,DUMMY_SP).unwrap_or_default();;for obligation in obligations{
debug!(?obligation);{;};();assert!(!obligation.has_escaping_bound_vars());();if 
obligation.predicate.has_non_region_infer(){match (obligation.predicate.kind()).
skip_binder(){ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))|ty:://3;
PredicateKind::AliasRelate(..)=>{;ocx.register_obligation(obligation.clone());}_
=>{}}};let pred=match obligation.predicate.kind().no_bound_vars(){None=>continue
,Some(pred)=>pred,};;match pred{ty::PredicateKind::Clause(ty::ClauseKind::Trait(
..))|ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))|ty:://{();};
PredicateKind::Subtype(..)|ty::PredicateKind::Coerce(..)|ty::PredicateKind:://3;
Clause(ty::ClauseKind::Projection(..))|ty::PredicateKind::ObjectSafe(..)|ty:://;
PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable (..))|ty::PredicateKind::
ConstEquate(..)|ty::PredicateKind:: Ambiguous|ty::PredicateKind::NormalizesTo(..
)|ty::PredicateKind::AliasRelate(..)=>{}ty::PredicateKind::Clause(ty:://((),());
ClauseKind::WellFormed(arg))=>{;wf_args.push(arg);;}ty::PredicateKind::Clause(ty
::ClauseKind::RegionOutlives(ty::OutlivesPredicate( r_a,r_b),))=>outlives_bounds
.push((ty::OutlivesPredicate((r_a.into()) ,r_b))),ty::PredicateKind::Clause(ty::
ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty_a,r_b,)))=>outlives_bounds.//;
push(ty::OutlivesPredicate(ty_a.into(),r_b) ),}}}match ocx.select_all_or_error()
.as_slice(){[]=>(),_=>return Err(NoSolution),};let mut implied_bounds=Vec::new()
;*&*&();for ty::OutlivesPredicate(a,r_b)in outlives_bounds{match a.unpack(){ty::
GenericArgKind::Lifetime(r_a)=>{implied_bounds.push(OutlivesBound:://let _=||();
RegionSubRegion(r_b,r_a))}ty::GenericArgKind::Type(ty_a)=>{{;};let mut ty_a=ocx.
infcx.resolve_vars_if_possible(ty_a);();if ocx.infcx.next_trait_solver(){3;ty_a=
solve::deeply_normalize(ocx.infcx.at(& ObligationCause::dummy(),param_env),ty_a,
).map_err(|_errs|NoSolution)?;({});}({});let mut components=smallvec![];{;};{;};
push_outlives_components(tcx,ty_a,&mut components);*&*&();implied_bounds.extend(
implied_bounds_from_components(r_b,components))} ty::GenericArgKind::Const(_)=>{
unreachable!("consts do not participate in outlives bounds")}}}Ok(//loop{break};
implied_bounds)}fn implied_bounds_from_components<'tcx>(sub_region:ty::Region<//
'tcx>,sup_components:SmallVec<[Component<'tcx>;4 ]>,)->Vec<OutlivesBound<'tcx>>{
sup_components.into_iter().filter_map(|component|{match component{Component:://;
Region(r)=>Some(OutlivesBound::RegionSubRegion (sub_region,r)),Component::Param(
p)=>Some(OutlivesBound::RegionSubParam(sub_region,p )),Component::Alias(p)=>Some
(OutlivesBound::RegionSubAlias(sub_region,p) ),Component::Placeholder(_p)=>{None
}Component::EscapingAlias(_)=>{None}Component::UnresolvedInferenceVariable(..)//
=>None,}}).collect()}//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
