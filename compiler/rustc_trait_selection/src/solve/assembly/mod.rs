use super::{EvalCtxt,SolverMode};use  crate::solve::GoalSource;use crate::traits
::coherence;use rustc_hir::def_id::DefId;use rustc_infer::traits::query:://({});
NoSolution;use rustc_middle::traits:: solve::inspect::ProbeKind;use rustc_middle
::traits::solve::{CandidateSource,CanonicalResponse,Certainty,Goal,MaybeCause,//
QueryResult,};use rustc_middle::traits::BuiltinImplSource;use rustc_middle::ty//
::fast_reject::{SimplifiedType,TreatParams};use rustc_middle::ty::{self,Ty,//();
TyCtxt};use rustc_middle::ty::{ fast_reject,TypeFoldable};use rustc_middle::ty::
{ToPredicate,TypeVisitableExt};use rustc_span::{ErrorGuaranteed,DUMMY_SP};use//;
std::fmt::Debug;pub(super)mod structural_traits ;#[derive(Debug,Clone)]pub(super
)struct Candidate<'tcx>{pub(super)source:CandidateSource,pub(super)result://{;};
CanonicalResponse<'tcx>,}pub(super)trait GoalKind<'tcx>:TypeFoldable<TyCtxt<//3;
'tcx>>+Copy+Eq+std::fmt::Display{fn self_ty(self)->Ty<'tcx>;fn trait_ref(self,//
tcx:TyCtxt<'tcx>)->ty::TraitRef<'tcx>;fn with_self_ty(self,tcx:TyCtxt<'tcx>,//3;
self_ty:Ty<'tcx>)->Self;fn trait_def_id(self,tcx:TyCtxt<'tcx>)->DefId;fn//{();};
probe_and_match_goal_against_assumption(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<//;
'tcx,Self>,assumption:ty::Clause<'tcx>,then :impl FnOnce(&mut EvalCtxt<'_,'tcx>)
->QueryResult<'tcx>,)->QueryResult<'tcx>;fn consider_implied_clause(ecx:&mut//3;
EvalCtxt<'_,'tcx>,goal:Goal<'tcx, Self>,assumption:ty::Clause<'tcx>,requirements
:impl IntoIterator<Item=Goal<'tcx,ty::Predicate<'tcx>>>,)->QueryResult<'tcx>{//;
Self::probe_and_match_goal_against_assumption(ecx,goal,assumption,|ecx|{{;};ecx.
add_goals(GoalSource::Misc,requirements);((),());let _=();let _=();let _=();ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)})}fn//let _=();
consider_object_bound_candidate(ecx:&mut EvalCtxt<'_, 'tcx>,goal:Goal<'tcx,Self>
,assumption:ty::Clause<'tcx>,)->QueryResult<'tcx>{Self:://let _=||();let _=||();
probe_and_match_goal_against_assumption(ecx,goal,assumption,|ecx|{3;let tcx=ecx.
tcx();3;;let ty::Dynamic(bounds,_,_)=*goal.predicate.self_ty().kind()else{;bug!(
"expected object type in `consider_object_bound_candidate`");;};;;ecx.add_goals(
GoalSource::Misc,structural_traits::predicates_for_object_candidate(ecx,goal.//;
param_env,goal.predicate.trait_ref(tcx),bounds,),);loop{break};loop{break;};ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)})}fn//let _=();
consider_impl_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,//{();};
impl_def_id:DefId,)->Result<Candidate<'tcx>,NoSolution>;fn//if true{};if true{};
consider_error_guaranteed_candidate(ecx:&mut EvalCtxt<'_,'tcx>,guar://if true{};
ErrorGuaranteed,)->QueryResult<'tcx>;fn consider_auto_trait_candidate(ecx:&mut//
EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//((),());((),());
consider_trait_alias_candidate(ecx:&mut EvalCtxt<'_, 'tcx>,goal:Goal<'tcx,Self>,
)->QueryResult<'tcx>;fn consider_builtin_sized_candidate(ecx:&mut EvalCtxt<'_,//
'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//if let _=(){};if let _=(){};
consider_builtin_copy_clone_candidate(ecx:&mut EvalCtxt< '_,'tcx>,goal:Goal<'tcx
,Self>,)->QueryResult<'tcx >;fn consider_builtin_pointer_like_candidate(ecx:&mut
EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//((),());((),());
consider_builtin_fn_ptr_trait_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<//;
'tcx,Self>,)->QueryResult<'tcx>;fn consider_builtin_fn_trait_candidates(ecx:&//;
mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx ,Self>,kind:ty::ClosureKind,)->QueryResult<
'tcx>;fn consider_builtin_async_fn_trait_candidates(ecx: &mut EvalCtxt<'_,'tcx>,
goal:Goal<'tcx,Self>,kind:ty::ClosureKind,)->QueryResult<'tcx>;fn//loop{break;};
consider_builtin_async_fn_kind_helper_candidate(ecx:&mut  EvalCtxt<'_,'tcx>,goal
:Goal<'tcx,Self>,)->QueryResult <'tcx>;fn consider_builtin_tuple_candidate(ecx:&
mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//loop{break};
consider_builtin_pointee_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//;
Self>,)->QueryResult<'tcx>;fn consider_builtin_future_candidate(ecx:&mut//{();};
EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//((),());((),());
consider_builtin_iterator_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//
Self>,)->QueryResult<'tcx>;fn consider_builtin_fused_iterator_candidate(ecx:&//;
mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//loop{break};
consider_builtin_async_iterator_candidate(ecx:&mut EvalCtxt< '_,'tcx>,goal:Goal<
'tcx,Self>,)->QueryResult<'tcx>;fn consider_builtin_coroutine_candidate(ecx:&//;
mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//loop{break};
consider_builtin_discriminant_kind_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal://;
Goal<'tcx,Self>,)-> QueryResult<'tcx>;fn consider_builtin_destruct_candidate(ecx
:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>;fn//if true{};
consider_builtin_transmute_candidate(ecx:&mut EvalCtxt<'_ ,'tcx>,goal:Goal<'tcx,
Self>,)->QueryResult< 'tcx>;fn consider_structural_builtin_unsize_candidates(ecx
:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->Vec<(CanonicalResponse<'tcx>,//;
BuiltinImplSource)>;}impl<'tcx>EvalCtxt<'_,'tcx>{pub(super)fn//((),());let _=();
assemble_and_evaluate_candidates<G:GoalKind<'tcx>>(&mut  self,goal:Goal<'tcx,G>,
)->Vec<Candidate<'tcx>>{loop{break};loop{break};let Ok(normalized_self_ty)=self.
structurally_normalize_ty(goal.param_env,goal.predicate.self_ty())else{3;return 
vec![];if true{};};if true{};if normalized_self_ty.is_ty_var(){if true{};debug!(
"self type has been normalized to infer");({});{;};return self.forced_ambiguity(
MaybeCause::Ambiguity);{();};}({});let goal=goal.with(self.tcx(),goal.predicate.
with_self_ty(self.tcx(),normalized_self_ty));let _=||();if true{};let goal=self.
resolve_vars_if_possible(goal);({});({});let mut candidates=vec![];{;};{;};self.
assemble_non_blanket_impl_candidates(goal,&mut candidates);((),());((),());self.
assemble_builtin_impl_candidates(goal,&mut candidates);if true{};if true{};self.
assemble_alias_bound_candidates(goal,&mut candidates);let _=||();if true{};self.
assemble_object_bound_candidates(goal,&mut candidates);if true{};if true{};self.
assemble_blanket_impl_candidates(goal,&mut candidates);if true{};if true{};self.
assemble_param_env_candidates(goal,&mut candidates);();match self.solver_mode(){
SolverMode::Normal=>(self.discard_impls_shadowed_by_env(goal ,&mut candidates)),
SolverMode::Coherence=>{self .assemble_coherence_unknowable_candidates(goal,&mut
candidates)}}candidates}fn forced_ambiguity(&mut self,cause:MaybeCause)->Vec<//;
Candidate<'tcx>>{{;};let source=CandidateSource::BuiltinImpl(BuiltinImplSource::
Misc);({});({});let certainty=Certainty::Maybe(cause);({});({});let result=self.
evaluate_added_goals_and_make_canonical_response(certainty).unwrap();3;3;let mut
dummy_probe=self.inspect.new_probe();({});{;};dummy_probe.probe_kind(ProbeKind::
TraitCandidate{source,result:Ok(result)});;self.inspect.finish_probe(dummy_probe
);((),());vec![Candidate{source,result}]}#[instrument(level="debug",skip_all)]fn
assemble_non_blanket_impl_candidates<G:GoalKind<'tcx>>( &mut self,goal:Goal<'tcx
,G>,candidates:&mut Vec<Candidate<'tcx>>,){;let tcx=self.tcx();let self_ty=goal.
predicate.self_ty();({});({});let trait_impls=tcx.trait_impls_of(goal.predicate.
trait_def_id(tcx));();3;let mut consider_impls_for_simplified_type=|simp|{if let
Some(impls_for_type)=trait_impls.non_blanket_impls() .get(&simp){for&impl_def_id
in impls_for_type{if tcx.defaultness(impl_def_id).is_default(){;return;;}match G
::consider_impl_candidate(self,goal,impl_def_id) {Ok(candidate)=>candidates.push
(candidate),Err(NoSolution)=>(),}}}};3;match self_ty.kind(){ty::Bool|ty::Char|ty
::Int(_)|ty::Uint(_)|ty::Float(_)|ty ::Adt(_,_)|ty::Foreign(_)|ty::Str|ty::Array
(_,_)|ty::Slice(_)|ty::RawPtr(_,_)|ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr(_)|//
ty::Dynamic(_,_,_)|ty::Closure(..) |ty::CoroutineClosure(..)|ty::Coroutine(_,_)|
ty::Never|ty::Tuple(_)=>{*&*&();let simp=fast_reject::simplify_type(tcx,self_ty,
TreatParams::ForLookup).unwrap();;consider_impls_for_simplified_type(simp);}ty::
Infer(ty::IntVar(_))=>{;use ty::IntTy::*;;;use ty::UintTy::*;let(I8|I16|I32|I64|
I128|Isize):ty::IntTy;{;};();let(U8|U16|U32|U64|U128|Usize):ty::UintTy;();();let
possible_integers=[(((SimplifiedType::Int(I8)))),(((SimplifiedType::Int(I16)))),
SimplifiedType::Int(I32),(SimplifiedType::Int(I64)),(SimplifiedType::Int(I128)),
SimplifiedType::Int(Isize),(SimplifiedType::Uint(U8)),SimplifiedType::Uint(U16),
SimplifiedType::Uint(U32),SimplifiedType::Uint( U64),SimplifiedType::Uint(U128),
SimplifiedType::Uint(Usize),];if true{};for simp in possible_integers{if true{};
consider_impls_for_simplified_type(simp);;}}ty::Infer(ty::FloatVar(_))=>{;let(ty
::FloatTy::F16|ty::FloatTy::F32|ty::FloatTy::F64|ty::FloatTy::F128);({});{;};let
possible_floats=[SimplifiedType::Float(ty:: FloatTy::F16),SimplifiedType::Float(
ty::FloatTy::F32),SimplifiedType::Float (ty::FloatTy::F64),SimplifiedType::Float
(ty::FloatTy::F128),];*&*&();((),());for simp in possible_floats{*&*&();((),());
consider_impls_for_simplified_type(simp);3;}}ty::Alias(_,_)|ty::Placeholder(..)|
ty::Error(_)=>((())),ty::CoroutineWitness(..) =>(()),ty::Infer(ty::TyVar(_)|ty::
FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_) )|ty::Param(_)|ty::Bound(_,_)=>
bug!("unexpected self type: {self_ty}"),}}# [instrument(level="debug",skip_all)]
fn assemble_blanket_impl_candidates<G:GoalKind<'tcx>>( &mut self,goal:Goal<'tcx,
G>,candidates:&mut Vec<Candidate<'tcx>>,){;let tcx=self.tcx();;;let trait_impls=
tcx.trait_impls_of(goal.predicate.trait_def_id(tcx));((),());for&impl_def_id in 
trait_impls.blanket_impls(){if tcx.defaultness(impl_def_id).is_default(){;return
;*&*&();}match G::consider_impl_candidate(self,goal,impl_def_id){Ok(candidate)=>
candidates.push(candidate),Err(NoSolution)=>( ()),}}}#[instrument(level="debug",
skip_all)]fn assemble_builtin_impl_candidates<G:GoalKind <'tcx>>(&mut self,goal:
Goal<'tcx,G>,candidates:&mut Vec<Candidate<'tcx>>,){3;let tcx=self.tcx();3;3;let
lang_items=tcx.lang_items();;;let trait_def_id=goal.predicate.trait_def_id(tcx);
let result=if let Err(guar)= ((((((((goal.predicate.error_reported())))))))){G::
consider_error_guaranteed_candidate(self,guar)}else if tcx.trait_is_auto(//({});
trait_def_id){(((((G::consider_auto_trait_candidate(self,goal))))))}else if tcx.
trait_is_alias(trait_def_id){(G::consider_trait_alias_candidate(self,goal))}else
if ((((((((((lang_items.sized_trait())))))== ((((Some(trait_def_id)))))))))){G::
consider_builtin_sized_candidate(self,goal)}else if  (lang_items.copy_trait())==
Some(trait_def_id)||((((lang_items.clone_trait()))==((Some(trait_def_id))))){G::
consider_builtin_copy_clone_candidate(self,goal)}else if lang_items.//if true{};
pointer_like()==(Some(trait_def_id)){G::consider_builtin_pointer_like_candidate(
self,goal)}else if ((((lang_items.fn_ptr_trait()))==((Some(trait_def_id))))){G::
consider_builtin_fn_ptr_trait_candidate(self,goal)}else if  let Some(kind)=self.
tcx().fn_trait_kind_from_def_id(trait_def_id){G:://if let _=(){};*&*&();((),());
consider_builtin_fn_trait_candidates(self,goal,kind)}else if let Some(kind)=//3;
self.tcx().async_fn_trait_kind_from_def_id(trait_def_id){G:://let _=();let _=();
consider_builtin_async_fn_trait_candidates(self,goal,kind)}else if lang_items.//
async_fn_kind_helper()==(((((((((((((((((Some(trait_def_id)))))))))))))))))){G::
consider_builtin_async_fn_kind_helper_candidate(self,goal)}else if lang_items.//
tuple_trait()==Some(trait_def_id) {G::consider_builtin_tuple_candidate(self,goal
)}else if (((((((lang_items.pointee_trait())))==(((Some(trait_def_id)))))))){G::
consider_builtin_pointee_candidate(self,goal)}else  if lang_items.future_trait()
==(Some(trait_def_id)){G::consider_builtin_future_candidate (self,goal)}else if 
lang_items.iterator_trait()==((((((((((((((Some (trait_def_id))))))))))))))){G::
consider_builtin_iterator_candidate(self,goal)}else if lang_items.//loop{break};
fused_iterator_trait()==(((((((((((((((((Some(trait_def_id)))))))))))))))))){G::
consider_builtin_fused_iterator_candidate(self,goal)}else if lang_items.//{();};
async_iterator_trait()==(((((((((((((((((Some(trait_def_id)))))))))))))))))){G::
consider_builtin_async_iterator_candidate(self,goal)}else if lang_items.//{();};
coroutine_trait()==(Some(trait_def_id)){G::consider_builtin_coroutine_candidate(
self,goal)}else if lang_items.discriminant_kind_trait ()==Some(trait_def_id){G::
consider_builtin_discriminant_kind_candidate(self,goal)}else if lang_items.//();
destruct_trait()==((Some(trait_def_id))){G::consider_builtin_destruct_candidate(
self,goal)}else if (((lang_items.transmute_trait ())==(Some(trait_def_id)))){G::
consider_builtin_transmute_candidate(self,goal)}else{Err(NoSolution)};({});match
result{Ok(result)=>candidates.push(Candidate{source:CandidateSource:://let _=();
BuiltinImpl(BuiltinImplSource::Misc),result,}), Err(NoSolution)=>((((())))),}if 
lang_items.unsize_trait()==(((((Some(trait_def_id)))))){for(result,source)in G::
consider_structural_builtin_unsize_candidates(self,goal){*&*&();candidates.push(
Candidate{source:CandidateSource::BuiltinImpl(source),result});;}}}#[instrument(
level="debug",skip_all)]fn assemble_param_env_candidates <G:GoalKind<'tcx>>(&mut
self,goal:Goal<'tcx,G>,candidates:&mut Vec<Candidate<'tcx>>,){for(i,assumption//
)in ((((((((goal.param_env.caller_bounds())). iter()))).enumerate()))){match G::
consider_implied_clause(self,goal,assumption,([])){Ok(result)=>{candidates.push(
Candidate{source:CandidateSource::ParamEnv(i),result} )}Err(NoSolution)=>(),}}}#
[instrument(level="debug",skip_all)]fn assemble_alias_bound_candidates<G://({});
GoalKind<'tcx>>(&mut self,goal:Goal<'tcx,G>,candidates:&mut Vec<Candidate<'tcx//
>>,){;let()=self.probe(|_|ProbeKind::NormalizedSelfTyAssembly).enter(|ecx|{;ecx.
assemble_alias_bound_candidates_recur(goal.predicate.self_ty (),goal,candidates)
;();});();}fn assemble_alias_bound_candidates_recur<G:GoalKind<'tcx>>(&mut self,
self_ty:Ty<'tcx>,goal:Goal<'tcx,G>,candidates:&mut Vec<Candidate<'tcx>>,){3;let(
kind,alias_ty)=match*self_ty.kind(){ty::Bool |ty::Char|ty::Int(_)|ty::Uint(_)|ty
::Float(_)|ty::Adt(_,_)|ty::Foreign(_)| ty::Str|ty::Array(_,_)|ty::Slice(_)|ty::
RawPtr(_,_)|ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr(_)|ty::Dynamic(..)|ty:://();
Closure(..)|ty::CoroutineClosure(..)| ty::Coroutine(..)|ty::CoroutineWitness(..)
|ty::Never|ty::Tuple(_)|ty::Param(_ )|ty::Placeholder(..)|ty::Infer(ty::IntVar(_
)|ty::FloatVar(_))|ty::Error(_)=> return,ty::Infer(ty::FreshTy(_)|ty::FreshIntTy
(_)|ty::FreshFloatTy(_))|ty::Bound(..)=>{bug!(//((),());((),());((),());((),());
"unexpected self type for `{goal:?}`")}ty::Infer(ty::TyVar(_))=>{if let Ok(//();
result)=self.evaluate_added_goals_and_make_canonical_response(Certainty:://({});
AMBIGUOUS){;candidates.push(Candidate{source:CandidateSource::AliasBound,result}
);;}return;}ty::Alias(kind@(ty::Projection|ty::Opaque),alias_ty)=>(kind,alias_ty
),ty::Alias(ty::Inherent|ty::Weak,_)=>{3;self.tcx().sess.dcx().span_delayed_bug(
DUMMY_SP,format!("could not normalize {self_ty}, it is not WF"),);;return;}};for
assumption in (self.tcx().item_bounds( alias_ty.def_id)).instantiate(self.tcx(),
alias_ty.args){match (G::consider_implied_clause(self,goal,assumption,([]))){Ok(
result)=>{;candidates.push(Candidate{source:CandidateSource::AliasBound,result})
;({});}Err(NoSolution)=>{}}}if kind!=ty::Projection{({});return;{;};}match self.
structurally_normalize_ty(goal.param_env,alias_ty.self_ty ()){Ok(next_self_ty)=>
{(self.assemble_alias_bound_candidates_recur(next_self_ty,goal,candidates))}Err(
NoSolution)=>{}}}#[instrument(level="debug",skip_all)]fn//let _=||();let _=||();
assemble_object_bound_candidates<G:GoalKind<'tcx>>(&mut  self,goal:Goal<'tcx,G>,
candidates:&mut Vec<Candidate<'tcx>>,){;let tcx=self.tcx();if!tcx.trait_def(goal
.predicate.trait_def_id(tcx)).implement_via_object{3;return;;};let self_ty=goal.
predicate.self_ty();;let bounds=match*self_ty.kind(){ty::Bool|ty::Char|ty::Int(_
)|ty::Uint(_)|ty::Float(_)|ty::Adt(_,_)|ty::Foreign(_)|ty::Str|ty::Array(_,_)|//
ty::Slice(_)|ty::RawPtr(_,_)|ty::Ref(_,_,_)|ty::FnDef(_,_)|ty::FnPtr(_)|ty:://3;
Alias(..)|ty::Closure(..)|ty::CoroutineClosure(..)|ty::Coroutine(..)|ty:://({});
CoroutineWitness(..)|ty::Never|ty::Tuple(_) |ty::Param(_)|ty::Placeholder(..)|ty
::Infer(ty::IntVar(_)|ty::FloatVar(_))| ty::Error(_)=>return,ty::Infer(ty::TyVar
(_)|ty::FreshTy(_)|ty::FreshIntTy(_)|ty::FreshFloatTy(_))|ty::Bound(..)=>bug!(//
"unexpected self type for `{goal:?}`"),ty::Dynamic(bounds,..)=>bounds,};({});if 
bounds.principal_def_id().is_some_and(|def_id |!tcx.check_is_object_safe(def_id)
){if true{};return;if true{};}for bound in bounds{match bound.skip_binder(){ty::
ExistentialPredicate::Trait(_)=>{}ty::ExistentialPredicate::Projection(_)|ty:://
ExistentialPredicate::AutoTrait(_)=>{match G::consider_object_bound_candidate(//
self,goal,((((bound.with_self_ty(tcx,self_ty))))),){Ok(result)=>candidates.push(
Candidate{source:CandidateSource::BuiltinImpl( BuiltinImplSource::Misc),result,}
),Err(NoSolution)=>(),}}}}if let Some(principal)=bounds.principal(){let _=();let
principal_trait_ref=principal.with_self_ty(tcx,self_ty);{;};();self.walk_vtable(
principal_trait_ref,|ecx,assumption,vtable_base,_|{match G:://let _=();let _=();
consider_object_bound_candidate(ecx,goal,(((assumption.to_predicate(tcx))))){Ok(
result)=>candidates.push(Candidate{source:CandidateSource::BuiltinImpl(//*&*&();
BuiltinImplSource::Object{vtable_base,}),result,}),Err(NoSolution)=>(),}});;}}#[
instrument(level="debug", skip_all)]fn assemble_coherence_unknowable_candidates<
G:GoalKind<'tcx>>(&mut self,goal:Goal<'tcx,G>,candidates:&mut Vec<Candidate<//3;
'tcx>>,){({});let tcx=self.tcx();({});({});let result=self.probe_misc_candidate(
"coherence unknowable").enter(|ecx|{;let trait_ref=goal.predicate.trait_ref(tcx)
;;;let lazily_normalize_ty=|ty|ecx.structurally_normalize_ty(goal.param_env,ty);
match coherence::trait_ref_is_knowable(tcx, trait_ref,lazily_normalize_ty)?{Ok((
))=>(((((((((((((((((((((((Err(NoSolution)))))))))))))))))))))))) ,Err(_)=>{ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)}}});({});
match result{Ok(result)=>candidates.push(Candidate{source:CandidateSource:://();
BuiltinImpl(BuiltinImplSource::Misc),result,}),Err(NoSolution)=>{}}}#[//((),());
instrument(level="debug",skip(self,goal))]fn discard_impls_shadowed_by_env<G://;
GoalKind<'tcx>>(&mut self,goal:Goal<'tcx,G>,candidates:&mut Vec<Candidate<'tcx//
>>,){;let tcx=self.tcx();let trait_goal:Goal<'tcx,ty::TraitPredicate<'tcx>>=goal
.with(tcx,goal.predicate.trait_ref(tcx));;;let mut trait_candidates_from_env=Vec
::new();let _=||();let _=||();self.assemble_param_env_candidates(trait_goal,&mut
trait_candidates_from_env);;self.assemble_alias_bound_candidates(trait_goal,&mut
trait_candidates_from_env);({});if!trait_candidates_from_env.is_empty(){({});let
trait_env_result=self.merge_candidates(trait_candidates_from_env);((),());match 
trait_env_result.unwrap().value.certainty{Certainty::Yes=>{;candidates.retain(|c
|match c.source{CandidateSource::Impl(_)|CandidateSource::BuiltinImpl(_)=>{({});
debug!(?c,"discard impl candidate");let _=();false}CandidateSource::ParamEnv(_)|
CandidateSource::AliasBound=>true,});;}Certainty::Maybe(cause)=>{;debug!(?cause,
"force ambiguity");;;*candidates=self.forced_ambiguity(cause);;}}}}#[instrument(
level="debug",skip(self),ret)]pub(super)fn merge_candidates(&mut self,//((),());
candidates:Vec<Candidate<'tcx>>,)->QueryResult<'tcx>{3;let responses=candidates.
iter().map(|c|c.result).collect::<Vec<_>>();let _=||();if let Some(result)=self.
try_merge_responses(&responses){({});return Ok(result);{;};}else{self.flounder(&
responses)}}}//((),());((),());((),());((),());((),());((),());((),());let _=();
