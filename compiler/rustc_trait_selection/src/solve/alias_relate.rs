use super::EvalCtxt;use rustc_infer::traits::query::NoSolution;use rustc_infer//
::traits::solve::GoalSource;use rustc_middle::traits::solve::{Certainty,Goal,//;
QueryResult};use rustc_middle::ty::{self,Ty};impl<'tcx>EvalCtxt<'_,'tcx>{#[//();
instrument(level="debug",skip(self) ,ret)]pub(super)fn compute_alias_relate_goal
(&mut self,goal:Goal<'tcx,(ty::Term<'tcx>,ty::Term<'tcx>,ty:://((),());let _=();
AliasRelationDirection)>,)->QueryResult<'tcx>{3;let tcx=self.tcx();3;3;let Goal{
param_env,predicate:(lhs,rhs,direction)}=goal;((),());*&*&();let Some(lhs)=self.
try_normalize_term(param_env,lhs)?else{if let _=(){};*&*&();((),());return self.
evaluate_added_goals_and_make_canonical_response(Certainty::overflow(true));;};;
let Some(rhs)=self.try_normalize_term(param_env,rhs)?else{if true{};return self.
evaluate_added_goals_and_make_canonical_response(Certainty::overflow(true));;};;
let variance=match direction{ty ::AliasRelationDirection::Equate=>ty::Variance::
Invariant,ty::AliasRelationDirection::Subtype=>ty::Variance::Covariant,};;match(
lhs.to_alias_ty(tcx),rhs.to_alias_ty(tcx)){(None,None)=>{;self.relate(param_env,
lhs,variance,rhs)?;*&*&();self.evaluate_added_goals_and_make_canonical_response(
Certainty::Yes)}(Some(alias),None)=>{self.relate_rigid_alias_non_alias(//*&*&();
param_env,alias,variance,rhs)}(None,Some(alias))=>self.//let _=||();loop{break};
relate_rigid_alias_non_alias(param_env,alias,variance.xform(ty::Variance:://{;};
Contravariant),lhs,),(Some(alias_lhs),Some(alias_rhs))=>{;self.relate(param_env,
alias_lhs,variance,alias_rhs)?;let _=||();let _=||();let _=||();let _=||();self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes) }}}#[instrument
(level="debug",skip(self,param_env),ret)]fn relate_rigid_alias_non_alias(&mut//;
self,param_env:ty::ParamEnv<'tcx>,alias :ty::AliasTy<'tcx>,variance:ty::Variance
,term:ty::Term<'tcx>,)->QueryResult<'tcx>{if term.is_infer(){;let tcx=self.tcx()
;;;let identity_args=self.fresh_args_for_item(alias.def_id);;let rigid_ctor=ty::
AliasTy::new(tcx,alias.def_id,identity_args);*&*&();((),());*&*&();((),());self.
eq_structurally_relating_aliases(param_env,term,rigid_ctor.to_ty(tcx).into())?;;
self.eq(param_env,alias,rigid_ctor)?;let _=();if true{};let _=();if true{};self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}else{Err(//{;};
NoSolution)}}#[instrument(level="debug",skip(self,param_env),ret)]fn//if true{};
try_normalize_term(&mut self,param_env:ty::ParamEnv< 'tcx>,term:ty::Term<'tcx>,)
->Result<Option<ty::Term<'tcx>>,NoSolution>{match (term.unpack()){ty::TermKind::
Ty(ty)=>{(Ok(self.try_normalize_ty_recur(param_env, 0,ty).map(Into::into)))}ty::
TermKind::Const(_)=>{if let Some(alias)=term.to_alias_ty(self.tcx()){3;let term=
self.next_term_infer_of_kind(term);;;self.add_normalizes_to_goal(Goal::new(self.
tcx(),param_env,ty::NormalizesTo{alias,term},));;self.try_evaluate_added_goals()
?;*&*&();Ok(Some(self.resolve_vars_if_possible(term)))}else{Ok(Some(term))}}}}#[
instrument(level="debug",skip(self,param_env),ret)]fn try_normalize_ty_recur(&//
mut self,param_env:ty::ParamEnv<'tcx>,depth: usize,ty:Ty<'tcx>,)->Option<Ty<'tcx
>>{if!self.tcx().recursion_limit().value_within_limit(depth){;return None;;};let
ty::Alias(kind,alias)=*ty.kind()else{;return Some(ty);};match self.commit_if_ok(
|this|{();let tcx=this.tcx();();3;let normalized_ty=this.next_ty_infer();3;3;let
normalizes_to=ty::NormalizesTo{alias,term:normalized_ty.into()};;match kind{ty::
AliasKind::Opaque=>{({});this.add_goal(GoalSource::Misc,Goal::new(tcx,param_env,
normalizes_to));let _=();}ty::AliasKind::Projection|ty::AliasKind::Inherent|ty::
AliasKind::Weak=>{this.add_normalizes_to_goal(Goal::new(tcx,param_env,//((),());
normalizes_to))}}let _=||();this.try_evaluate_added_goals()?;let _=||();Ok(this.
resolve_vars_if_possible(normalized_ty))}) {Ok(ty)=>self.try_normalize_ty_recur(
param_env,(((((depth+(((((1)))))))))),ty), Err(NoSolution)=>((((Some(ty))))),}}}
