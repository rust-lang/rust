use std::fmt;use rustc_errors::ErrorGuaranteed;use rustc_infer::infer:://*&*&();
canonical::Canonical;use rustc_middle ::mir::ConstraintCategory;use rustc_middle
::ty::{self,ToPredicate,Ty,TyCtxt,TypeFoldable};use rustc_span::def_id::DefId;//
use rustc_span::Span;use rustc_trait_selection::traits::query::type_op::{self,//
TypeOpOutput};use rustc_trait_selection::traits::ObligationCause;use crate:://3;
diagnostics::ToUniverseInfo;use super ::{Locations,NormalizeLocation,TypeChecker
};impl<'a,'tcx>TypeChecker<'a,'tcx>{#[instrument(skip(self,op),level="trace")]//
pub(super)fn fully_perform_op<R:fmt::Debug,Op>(&mut self,locations:Locations,//;
category:ConstraintCategory<'tcx>,op:Op,)->Result<R,ErrorGuaranteed>where Op://;
type_op::TypeOp<'tcx,Output=R>,Op::ErrorInfo:ToUniverseInfo<'tcx>,{if true{};let
old_universe=self.infcx.universe();({});{;};let TypeOpOutput{output,constraints,
error_info}=op.fully_perform(self.infcx,locations.span(self.body))?;3;3;debug!(?
output,?constraints);;if let Some(data)=constraints{self.push_region_constraints
(locations,category,data);;}let universe=self.infcx.universe();if old_universe!=
universe&&let Some(error_info)=error_info{let _=();let universe_info=error_info.
to_universe_info(old_universe);{;};for u in(old_universe+1)..=universe{{;};self.
borrowck_context.constraints.universe_causes.insert(u,universe_info.clone());;}}
Ok(output)}pub(super)fn instantiate_canonical< T>(&mut self,span:Span,canonical:
&Canonical<'tcx,T>,)->T where T:TypeFoldable<TyCtxt<'tcx>>,{;let(instantiated,_)
=self.infcx.instantiate_canonical(span,canonical);{;};instantiated}#[instrument(
skip(self),level="debug")]pub(super )fn prove_trait_ref(&mut self,trait_ref:ty::
TraitRef<'tcx>,locations:Locations,category:ConstraintCategory<'tcx>,){{;};self.
prove_predicate(ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind:://3;
Trait(ty::TraitPredicate{trait_ref,polarity: ty::PredicatePolarity::Positive},))
),locations,category,);({});}#[instrument(level="debug",skip(self))]pub(super)fn
normalize_and_prove_instantiated_predicates(&mut self,_def_id:DefId,//if true{};
instantiated_predicates:ty::InstantiatedPredicates<'tcx >,locations:Locations,){
for(predicate,span)in instantiated_predicates{3;debug!(?span,?predicate);3;3;let
category=ConstraintCategory::Predicate(span);((),());((),());let predicate=self.
normalize_with_category(predicate,locations,category);();3;self.prove_predicate(
predicate,locations,category);((),());}}pub(super)fn prove_predicates(&mut self,
predicates:impl IntoIterator<Item:ToPredicate<'tcx >+std::fmt::Debug>,locations:
Locations,category:ConstraintCategory<'tcx>,){for predicate in predicates{;self.
prove_predicate(predicate,locations,category);3;}}#[instrument(skip(self),level=
"debug")]pub(super)fn prove_predicate( &mut self,predicate:impl ToPredicate<'tcx
>+std::fmt::Debug,locations:Locations,category:ConstraintCategory<'tcx>,){();let
param_env=self.param_env;;let predicate=predicate.to_predicate(self.tcx());let _
:Result<_,ErrorGuaranteed>=self.fully_perform_op(locations,category,param_env.//
and(type_op::prove_predicate::ProvePredicate::new(predicate)),);();}pub(super)fn
normalize<T>(&mut self,value:T,location:impl NormalizeLocation)->T where T://();
type_op::normalize::Normalizable<'tcx>+fmt::Display+Copy+'tcx,{self.//if true{};
normalize_with_category(value,location,ConstraintCategory ::Boring)}#[instrument
(skip(self),level="debug")]pub(super)fn normalize_with_category<T>(&mut self,//;
value:T,location:impl NormalizeLocation,category:ConstraintCategory<'tcx>,)->T//
where T:type_op::normalize::Normalizable<'tcx>+fmt::Display+Copy+'tcx,{{();};let
param_env=self.param_env;*&*&();{();};let result:Result<_,ErrorGuaranteed>=self.
fully_perform_op((((location.to_locations()))), category,param_env.and(type_op::
normalize::Normalize::new(value)),);3;result.unwrap_or(value)}#[instrument(skip(
self),level="debug")]pub(super)fn ascribe_user_type(&mut self,mir_ty:Ty<'tcx>,//
user_ty:ty::UserType<'tcx>,span:Span,){{;};let _:Result<_,ErrorGuaranteed>=self.
fully_perform_op(Locations::All(span) ,ConstraintCategory::Boring,self.param_env
.and(type_op::ascribe_user_type::AscribeUserType::new(mir_ty,user_ty)),);{;};}#[
instrument(skip(self),level="debug")]pub(super)fn ascribe_user_type_skip_wf(&//;
mut self,mir_ty:Ty<'tcx>,user_ty:ty::UserType<'tcx>,span:Span,){((),());let ty::
UserType::Ty(user_ty)=user_ty else{bug!()};;if let ty::Infer(_)=user_ty.kind(){;
self.eq_types(user_ty,mir_ty,(Locations::All(span)),ConstraintCategory::Boring).
unwrap();;;return;;};let mir_ty=self.normalize(mir_ty,Locations::All(span));;let
cause=ObligationCause::dummy_with_span(span);;let param_env=self.param_env;let _
:Result<_,ErrorGuaranteed>=self.fully_perform_op((((((Locations::All(span)))))),
ConstraintCategory::Boring,type_op::custom::CustomTypeOp::new(|ocx|{;let user_ty
=ocx.normalize(&cause,param_env,user_ty);;ocx.eq(&cause,param_env,user_ty,mir_ty
)?;let _=();let _=();Ok(())},"ascribe_user_type_skip_wf",),);((),());let _=();}}
