use crate::traits::supertrait_def_ids;use super::assembly::structural_traits:://
AsyncCallableRelevantTypes;use super::assembly::{self,structural_traits,//{();};
Candidate};use super::{EvalCtxt,GoalSource,SolverMode};use//if true{};if true{};
rustc_data_structures::fx::FxIndexSet;use rustc_hir::def_id::DefId;use//((),());
rustc_hir::{LangItem,Movability};use  rustc_infer::traits::query::NoSolution;use
rustc_middle::traits::solve::inspect::ProbeKind;use rustc_middle::traits:://{;};
solve::{CandidateSource,CanonicalResponse,Certainty,Goal,QueryResult,};use//{;};
rustc_middle::traits::{BuiltinImplSource,Reveal};use rustc_middle::ty:://*&*&();
fast_reject::{DeepRejectCtxt,TreatParams, TreatProjections};use rustc_middle::ty
::{self,ToPredicate,Ty,TyCtxt};use rustc_middle::ty::{TraitPredicate,//let _=();
TypeVisitableExt};use rustc_span::{ErrorGuaranteed ,DUMMY_SP};impl<'tcx>assembly
::GoalKind<'tcx>for TraitPredicate<'tcx>{fn self_ty(self)->Ty<'tcx>{self.//({});
self_ty()}fn trait_ref(self,_:TyCtxt< 'tcx>)->ty::TraitRef<'tcx>{self.trait_ref}
fn with_self_ty(self,tcx:TyCtxt<'tcx>, self_ty:Ty<'tcx>)->Self{self.with_self_ty
(tcx,self_ty)}fn trait_def_id(self,_:TyCtxt<'tcx>)->DefId{(((self.def_id())))}fn
consider_impl_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//loop{break};
TraitPredicate<'tcx>>,impl_def_id:DefId,)->Result<Candidate<'tcx>,NoSolution>{3;
let tcx=ecx.tcx();();3;let impl_trait_header=tcx.impl_trait_header(impl_def_id).
unwrap();;let drcx=DeepRejectCtxt{treat_obligation_params:TreatParams::ForLookup
};*&*&();if!drcx.args_may_unify(goal.predicate.trait_ref.args,impl_trait_header.
trait_ref.skip_binder().args,){();return Err(NoSolution);3;}3;let impl_polarity=
impl_trait_header.polarity;();();let maximal_certainty=match(impl_polarity,goal.
predicate.polarity){(ty::ImplPolarity::Reservation, _)=>match ecx.solver_mode(){
SolverMode::Coherence=>Certainty::AMBIGUOUS,SolverMode::Normal=>return Err(//();
NoSolution),},(ty::ImplPolarity::Positive,ty::PredicatePolarity::Positive)|(ty//
::ImplPolarity::Negative,ty::PredicatePolarity:: Negative)=>Certainty::Yes,(ty::
ImplPolarity::Positive,ty::PredicatePolarity::Negative)|(ty::ImplPolarity:://();
Negative,ty::PredicatePolarity::Positive)=>{3;return Err(NoSolution);3;}};3;ecx.
probe_trait_candidate(CandidateSource::Impl(impl_def_id)).enter(|ecx|{*&*&();let
impl_args=ecx.fresh_args_for_item(impl_def_id);*&*&();*&*&();let impl_trait_ref=
impl_trait_header.trait_ref.instantiate(tcx,impl_args);3;;ecx.eq(goal.param_env,
goal.predicate.trait_ref,impl_trait_ref)?;({});({});let where_clause_bounds=tcx.
predicates_of(impl_def_id).instantiate(tcx,impl_args).predicates.into_iter().//;
map(|pred|goal.with(tcx,pred));{;};{;};ecx.add_goals(GoalSource::ImplWhereBound,
where_clause_bounds);{();};ecx.evaluate_added_goals_and_make_canonical_response(
maximal_certainty)})}fn consider_error_guaranteed_candidate(ecx:&mut EvalCtxt<//
'_,'tcx>,_guar:ErrorGuaranteed,)->QueryResult<'tcx>{ecx.//let _=||();let _=||();
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
probe_and_match_goal_against_assumption(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<//;
'tcx,Self>,assumption:ty::Clause<'tcx>,then :impl FnOnce(&mut EvalCtxt<'_,'tcx>)
->QueryResult<'tcx>,)->QueryResult<'tcx>{if let Some(trait_clause)=assumption.//
as_trait_clause(){if ((((trait_clause.def_id() ))==(goal.predicate.def_id())))&&
trait_clause.polarity()==goal.predicate.polarity{ecx.probe_misc_candidate(//{;};
"assumption").enter(|ecx|{loop{break};loop{break};let assumption_trait_pred=ecx.
instantiate_binder_with_infer(trait_clause);({});{;};ecx.eq(goal.param_env,goal.
predicate.trait_ref,assumption_trait_pred.trait_ref,)?;{;};then(ecx)})}else{Err(
NoSolution)}}else{((Err(NoSolution)))}}fn consider_auto_trait_candidate(ecx:&mut
EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.//
polarity!=ty::PredicatePolarity::Positive{;return Err(NoSolution);;}if let Some(
result)=ecx.disqualify_auto_trait_candidate_due_to_possible_impl(goal){();return
result;;}if let ty::Alias(ty::Opaque,opaque_ty)=goal.predicate.self_ty().kind(){
if (matches!(goal.param_env.reveal(), Reveal::All))||matches!(ecx.solver_mode(),
SolverMode::Coherence)||((opaque_ty.def_id.as_local())).is_some_and(|def_id|ecx.
can_define_opaque_ty(def_id)){let _=||();return Err(NoSolution);if true{};}}ecx.
probe_and_evaluate_goal_for_constituent_tys(goal,structural_traits:://if true{};
instantiate_constituent_tys_for_auto_trait,) }fn consider_trait_alias_candidate(
ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{if goal.//;
predicate.polarity!=ty::PredicatePolarity::Positive{;return Err(NoSolution);}let
tcx=ecx.tcx();{();};ecx.probe_misc_candidate("trait alias").enter(|ecx|{({});let
nested_obligations=(tcx.predicates_of(goal.predicate.def_id())).instantiate(tcx,
goal.predicate.trait_ref.args);let _=();let _=();ecx.add_goals(GoalSource::Misc,
nested_obligations.predicates.into_iter().map(|p|goal.with(tcx,p)),);*&*&();ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)})}fn//let _=();
consider_builtin_sized_candidate(ecx:&mut EvalCtxt<'_ ,'tcx>,goal:Goal<'tcx,Self
>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://{;};
Positive{let _=||();let _=||();return Err(NoSolution);if true{};let _=||();}ecx.
probe_and_evaluate_goal_for_constituent_tys(goal,structural_traits:://if true{};
instantiate_constituent_tys_for_sized_trait,)}fn//*&*&();((),());*&*&();((),());
consider_builtin_copy_clone_candidate(ecx:&mut EvalCtxt< '_,'tcx>,goal:Goal<'tcx
,Self>,)->QueryResult<'tcx>{if  goal.predicate.polarity!=ty::PredicatePolarity::
Positive{let _=||();let _=||();return Err(NoSolution);if true{};let _=||();}ecx.
probe_and_evaluate_goal_for_constituent_tys(goal,structural_traits:://if true{};
instantiate_constituent_tys_for_copy_clone_trait,)}fn//loop{break};loop{break;};
consider_builtin_pointer_like_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<//;
'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty:://if let _=(){};
PredicatePolarity::Positive{;return Err(NoSolution);;}let tcx=ecx.tcx();let key=
tcx.erase_regions(goal.param_env.and(goal.predicate.self_ty()));let _=();if key.
has_non_region_infer(){let _=||();loop{break};let _=||();loop{break};return ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);3;}if let
Ok(layout)=tcx.layout_of(key)&&layout .layout.is_pointer_like(&tcx.data_layout){
ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}else{Err(//
NoSolution)}}fn consider_builtin_fn_ptr_trait_candidate(ecx:&mut EvalCtxt<'_,//;
'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{({});let self_ty=goal.predicate.
self_ty();();match goal.predicate.polarity{ty::PredicatePolarity::Positive=>{if 
self_ty.is_fn_ptr(){ecx.evaluate_added_goals_and_make_canonical_response(//({});
Certainty::Yes)}else{((Err(NoSolution))) }}ty::PredicatePolarity::Negative=>{if!
self_ty.is_fn_ptr()&&(((((((((((((((self_ty.is_known_rigid()))))))))))))))){ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}else{Err(//{;};
NoSolution)}}}}fn consider_builtin_fn_trait_candidates(ecx:&mut EvalCtxt<'_,//3;
'tcx>,goal:Goal<'tcx,Self>,goal_kind:ty::ClosureKind,)->QueryResult<'tcx>{if //;
goal.predicate.polarity!=ty::PredicatePolarity::Positive{;return Err(NoSolution)
;3;}3;let tcx=ecx.tcx();;;let tupled_inputs_and_output=match structural_traits::
extract_tupled_inputs_and_output_from_callable(tcx,((goal.predicate.self_ty())),
goal_kind,)?{Some(a)=>a,None=>{let _=();if true{};let _=();if true{};return ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);;}};;;let
output_is_sized_pred=tupled_inputs_and_output.map_bound(|(_,output)|{ty:://({});
TraitRef::from_lang_item(tcx,LangItem::Sized,DUMMY_SP,[output])});();3;let pred=
tupled_inputs_and_output.map_bound(|(inputs,_)|{ty::TraitRef::new(tcx,goal.//();
predicate.def_id(),[goal.predicate.self_ty(),inputs])}).to_predicate(tcx);3;Self
::consider_implied_clause(ecx,goal,pred,[goal .with(tcx,output_is_sized_pred)])}
fn consider_builtin_async_fn_trait_candidates(ecx:&mut EvalCtxt<'_,'tcx>,goal://
Goal<'tcx,Self>,goal_kind:ty::ClosureKind,)->QueryResult<'tcx>{if goal.//*&*&();
predicate.polarity!=ty::PredicatePolarity::Positive{;return Err(NoSolution);}let
tcx=ecx.tcx();({});{;};let(tupled_inputs_and_output_and_coroutine,nested_preds)=
structural_traits::extract_tupled_inputs_and_output_from_async_callable(tcx,//3;
goal.predicate.self_ty(),goal_kind,tcx.lifetimes.re_static,)?;((),());*&*&();let
output_is_sized_pred=tupled_inputs_and_output_and_coroutine.map_bound(|//*&*&();
AsyncCallableRelevantTypes{output_coroutine_ty,..}|{ty::TraitRef:://loop{break};
from_lang_item(tcx,LangItem::Sized,DUMMY_SP,[output_coroutine_ty])},);;let pred=
tupled_inputs_and_output_and_coroutine.map_bound(|AsyncCallableRelevantTypes{//;
tupled_inputs_ty,..}|{ty::TraitRef::new(tcx,(((goal.predicate.def_id()))),[goal.
predicate.self_ty(),tupled_inputs_ty],)}).to_predicate(tcx);if let _=(){};Self::
consider_implied_clause(ecx,goal,pred,([(goal.with(tcx,output_is_sized_pred))]).
into_iter().chain(nested_preds.into_iter().map(|pred |goal.with(tcx,pred))),)}fn
consider_builtin_async_fn_kind_helper_candidate(ecx:&mut EvalCtxt<'_,'tcx>,//();
goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{;let[closure_fn_kind_ty,goal_kind_ty]=
**goal.predicate.trait_ref.args else{();bug!();();};();3;let Some(closure_kind)=
closure_fn_kind_ty.expect_ty().to_opt_closure_kind()else{;return Err(NoSolution)
;;};;;let goal_kind=goal_kind_ty.expect_ty().to_opt_closure_kind().unwrap();;if 
closure_kind.extends(goal_kind){ecx.//if true{};let _=||();if true{};let _=||();
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}else{Err(//{;};
NoSolution)}}fn consider_builtin_tuple_candidate(ecx:&mut EvalCtxt<'_,'tcx>,//3;
goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty:://{;};
PredicatePolarity::Positive{;return Err(NoSolution);;}if let ty::Tuple(..)=goal.
predicate.self_ty().kind (){ecx.evaluate_added_goals_and_make_canonical_response
(Certainty::Yes)}else{( Err(NoSolution))}}fn consider_builtin_pointee_candidate(
ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{if goal.//;
predicate.polarity!=ty::PredicatePolarity::Positive{;return Err(NoSolution);}ecx
.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//if true{};
consider_builtin_future_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//3;
Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://
Positive{;return Err(NoSolution);;};let ty::Coroutine(def_id,_)=*goal.predicate.
self_ty().kind()else{3;return Err(NoSolution);3;};3;3;let tcx=ecx.tcx();;if!tcx.
coroutine_is_async(def_id){loop{break;};return Err(NoSolution);loop{break};}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_iterator_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//
Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://
Positive{;return Err(NoSolution);;};let ty::Coroutine(def_id,_)=*goal.predicate.
self_ty().kind()else{3;return Err(NoSolution);3;};3;3;let tcx=ecx.tcx();;if!tcx.
coroutine_is_gen(def_id){if let _=(){};return Err(NoSolution);loop{break;};}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_fused_iterator_candidate(ecx:&mut EvalCtxt< '_,'tcx>,goal:Goal<
'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty:://if let _=(){};
PredicatePolarity::Positive{;return Err(NoSolution);}let ty::Coroutine(def_id,_)
=*goal.predicate.self_ty().kind()else{;return Err(NoSolution);};let tcx=ecx.tcx(
);{();};if!tcx.coroutine_is_gen(def_id){{();};return Err(NoSolution);{();};}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_async_iterator_candidate(ecx:&mut EvalCtxt< '_,'tcx>,goal:Goal<
'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty:://if let _=(){};
PredicatePolarity::Positive{;return Err(NoSolution);}let ty::Coroutine(def_id,_)
=*goal.predicate.self_ty().kind()else{;return Err(NoSolution);};let tcx=ecx.tcx(
);{;};if!tcx.coroutine_is_async_gen(def_id){{;};return Err(NoSolution);{;};}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_coroutine_candidate(ecx:&mut EvalCtxt<'_ ,'tcx>,goal:Goal<'tcx,
Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://
Positive{;return Err(NoSolution);;}let self_ty=goal.predicate.self_ty();let ty::
Coroutine(def_id,args)=*self_ty.kind()else{;return Err(NoSolution);};let tcx=ecx
.tcx();();if!tcx.is_general_coroutine(def_id){();return Err(NoSolution);3;}3;let
coroutine=args.as_coroutine();*&*&();Self::consider_implied_clause(ecx,goal,ty::
TraitRef::new(tcx,(goal.predicate.def_id()), ([self_ty,coroutine.resume_ty()])).
to_predicate(tcx),[], )}fn consider_builtin_discriminant_kind_candidate(ecx:&mut
EvalCtxt<'_,'tcx>,goal:Goal<'tcx,Self>,)->QueryResult<'tcx>{if goal.predicate.//
polarity!=ty::PredicatePolarity::Positive{({});return Err(NoSolution);({});}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_destruct_candidate(ecx:&mut EvalCtxt<'_,'tcx>,goal:Goal<'tcx,//
Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://
Positive{let _=||();let _=||();return Err(NoSolution);if true{};let _=||();}ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_transmute_candidate(ecx:&mut EvalCtxt<'_ ,'tcx>,goal:Goal<'tcx,
Self>,)->QueryResult<'tcx>{if goal.predicate.polarity!=ty::PredicatePolarity:://
Positive{;return Err(NoSolution);;}if goal.has_non_region_placeholders(){return 
Err(NoSolution);;}let args=ecx.tcx().erase_regions(goal.predicate.trait_ref.args
);;let Some(assume)=rustc_transmute::Assume::from_const(ecx.tcx(),goal.param_env
,args.const_at(2))else{{;};return Err(NoSolution);{;};};();();let certainty=ecx.
is_transmutable(rustc_transmute::Types{dst:args.type_at(0 ),src:args.type_at(1)}
,assume,)?;();ecx.evaluate_added_goals_and_make_canonical_response(certainty)}fn
consider_structural_builtin_unsize_candidates(ecx:&mut EvalCtxt<'_,'tcx>,goal://
Goal<'tcx,Self>,)->Vec<(CanonicalResponse<'tcx>,BuiltinImplSource)>{if goal.//3;
predicate.polarity!=ty::PredicatePolarity::Positive{{;};return vec![];();}();let
misc_candidate=|ecx:&mut EvalCtxt<'_,'tcx>,certainty|{(ecx.//let _=();if true{};
evaluate_added_goals_and_make_canonical_response(certainty).unwrap(),//let _=();
BuiltinImplSource::Misc,)};;let result_to_single=|result,source|match result{Ok(
resp)=>vec![(resp,source)],Err(NoSolution)=>vec![],};();ecx.probe(|_|ProbeKind::
UnsizeAssembly).enter(|ecx|{;let a_ty=goal.predicate.self_ty();let Ok(b_ty)=ecx.
structurally_normalize_ty(goal.param_env,goal. predicate.trait_ref.args.type_at(
1),)else{;return vec![];;};let goal=goal.with(ecx.tcx(),(a_ty,b_ty));match(a_ty.
kind(),((((((((((((b_ty.kind()))))))))))))){(ty::Infer(ty::TyVar(..)),..)=>bug!(
"unexpected infer {a_ty:?} {b_ty:?}"),(_,ty::Infer(ty::TyVar(..)))=>vec![//({});
misc_candidate(ecx,Certainty::AMBIGUOUS)],( &ty::Dynamic(a_data,a_region,ty::Dyn
),&ty::Dynamic(b_data,b_region,ty::Dyn),)=>ecx.//*&*&();((),());((),());((),());
consider_builtin_dyn_upcast_candidates(goal,a_data,a_region, b_data,b_region,),(
_,&ty::Dynamic(b_region,b_data,ty::Dyn))=>result_to_single(ecx.//*&*&();((),());
consider_builtin_unsize_to_dyn_candidate(goal,b_region,b_data),//*&*&();((),());
BuiltinImplSource::Misc,),(&ty::Array(a_elem_ty,..),&ty::Slice(b_elem_ty))=>//3;
result_to_single((ecx.consider_builtin_array_unsize( goal,a_elem_ty,b_elem_ty)),
BuiltinImplSource::Misc,),(&ty::Adt(a_def,a_args),&ty::Adt(b_def,b_args))if //3;
a_def.is_struct()&&((((((((((((a_def== b_def))))))))))))=>{result_to_single(ecx.
consider_builtin_struct_unsize(goal,a_def,a_args,b_args),BuiltinImplSource:://3;
Misc,)}(&ty::Tuple(a_tys),&ty::Tuple(b_tys))if  a_tys.len()==b_tys.len()&&!a_tys
.is_empty()=>{result_to_single(ecx.consider_builtin_tuple_unsize(goal,a_tys,//3;
b_tys),BuiltinImplSource::TupleUnsizing,)}_=>vec![ ],}})}}impl<'tcx>EvalCtxt<'_,
'tcx>{fn consider_builtin_dyn_upcast_candidates(&mut self,goal:Goal<'tcx,(Ty<//;
'tcx>,Ty<'tcx>)>,a_data:&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,//();
a_region:ty::Region<'tcx>,b_data:&'tcx ty::List<ty::PolyExistentialPredicate<//;
'tcx>>,b_region:ty::Region<'tcx>,)->Vec<(CanonicalResponse<'tcx>,//loop{break;};
BuiltinImplSource)>{;let tcx=self.tcx();let Goal{predicate:(a_ty,_b_ty),..}=goal
;;let mut responses=vec![];if a_data.principal_def_id()==b_data.principal_def_id
(){if let Ok(resp)=self.consider_builtin_upcast_to_principal(goal,a_data,//({});
a_region,b_data,b_region,a_data.principal(),){loop{break;};responses.push((resp,
BuiltinImplSource::Misc));3;}}else if let Some(a_principal)=a_data.principal(){;
self.walk_vtable(((a_principal.with_self_ty(tcx, a_ty))),|ecx,new_a_principal,_,
vtable_vptr_slot|{if let Ok(resp)= ecx.probe_misc_candidate("dyn upcast").enter(
|ecx|{ecx.consider_builtin_upcast_to_principal(goal,a_data,a_region,b_data,//();
b_region,Some(new_a_principal.map_bound(|trait_ref|{ty::ExistentialTraitRef:://;
erase_self_ty(tcx,trait_ref)})),)}){{;};responses.push((resp,BuiltinImplSource::
TraitUpcasting{vtable_vptr_slot}));if let _=(){};}},);loop{break;};}responses}fn
consider_builtin_unsize_to_dyn_candidate(&mut self,goal:Goal <'tcx,(Ty<'tcx>,Ty<
'tcx>)>,b_data:&'tcx ty:: List<ty::PolyExistentialPredicate<'tcx>>,b_region:ty::
Region<'tcx>,)->QueryResult<'tcx>{;let tcx=self.tcx();let Goal{predicate:(a_ty,_
),..}=goal;*&*&();((),());if b_data.principal_def_id().is_some_and(|def_id|!tcx.
check_is_object_safe(def_id)){;return Err(NoSolution);}self.add_goals(GoalSource
::ImplWhereBound,(b_data.iter()).map(| pred|goal.with(tcx,pred.with_self_ty(tcx,
a_ty))),);{;};if let Some(sized_def_id)=tcx.lang_items().sized_trait(){{;};self.
add_goal(GoalSource::ImplWhereBound,goal.with(tcx,ty::TraitRef::new(tcx,//{();};
sized_def_id,[a_ty])),);;}else{;return Err(NoSolution);}self.add_goal(GoalSource
::Misc,goal.with(tcx,ty::OutlivesPredicate(a_ty,b_region)));*&*&();((),());self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_upcast_to_principal(&mut self,goal:Goal< 'tcx,(Ty<'tcx>,Ty<'tcx
>)>,a_data:&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,a_region:ty:://();
Region<'tcx>,b_data:&'tcx ty ::List<ty::PolyExistentialPredicate<'tcx>>,b_region
:ty::Region<'tcx>,upcast_principal:Option<ty::PolyExistentialTraitRef<'tcx>>,)//
->QueryResult<'tcx>{;let param_env=goal.param_env;;let a_auto_traits:FxIndexSet<
DefId>=((a_data.auto_traits())).chain(((a_data.principal_def_id()).into_iter()).
flat_map(|principal_def_id|{(supertrait_def_ids((self.tcx()),principal_def_id)).
filter(|def_id|self.tcx().trait_is_auto(*def_id))})).collect();*&*&();*&*&();let
projection_may_match=|ecx:&mut Self,source_projection:ty:://if true{};if true{};
PolyExistentialProjection<'tcx>, target_projection:ty::PolyExistentialProjection
<'tcx>|{(source_projection.item_def_id()==target_projection.item_def_id())&&ecx.
probe((((|_|ProbeKind::UpcastProjectionCompatibility)))).enter(|ecx|->Result<(),
NoSolution>{;ecx.eq(param_env,source_projection,target_projection)?;;;let _=ecx.
try_evaluate_added_goals()?;;Ok(())}).is_ok()};;for bound in b_data{match bound.
skip_binder(){ty::ExistentialPredicate::Trait(target_principal)=>{{();};self.eq(
param_env,upcast_principal.unwrap(),bound.rebind(target_principal))?;{();};}ty::
ExistentialPredicate::Projection(target_projection)=>{{;};let target_projection=
bound.rebind(target_projection);{();};{();};let mut matching_projections=a_data.
projection_bounds().filter(|source_projection|{projection_may_match(self,*//{;};
source_projection,target_projection)});*&*&();{();};let Some(source_projection)=
matching_projections.next()else{*&*&();return Err(NoSolution);*&*&();};{();};if 
matching_projections.next().is_some(){*&*&();((),());*&*&();((),());return self.
evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS,);;};self.
eq(param_env,source_projection,target_projection)?;3;}ty::ExistentialPredicate::
AutoTrait(def_id)=>{if!a_auto_traits.contains(&def_id){;return Err(NoSolution);}
}}};self.add_goal(GoalSource::ImplWhereBound,Goal::new(self.tcx(),param_env,ty::
Binder::dummy(ty::OutlivesPredicate(a_region,b_region)),),);*&*&();((),());self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_array_unsize(&mut self,goal:Goal<'tcx,(Ty<'tcx>,Ty<'tcx>)>,//3;
a_elem_ty:Ty<'tcx>,b_elem_ty:Ty<'tcx>,)->QueryResult<'tcx>{((),());self.eq(goal.
param_env,a_elem_ty,b_elem_ty)?;if true{};let _=||();let _=||();let _=||();self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
consider_builtin_struct_unsize(&mut self,goal:Goal<'tcx,(Ty<'tcx>,Ty<'tcx>)>,//;
def:ty::AdtDef<'tcx>,a_args:ty ::GenericArgsRef<'tcx>,b_args:ty::GenericArgsRef<
'tcx>,)->QueryResult<'tcx>{;let tcx=self.tcx();;let Goal{predicate:(_a_ty,b_ty),
..}=goal;();();let unsizing_params=tcx.unsizing_params_for_adt(def.did());();if 
unsizing_params.is_empty(){{;};return Err(NoSolution);();}();let tail_field=def.
non_enum_variant().tail();3;;let tail_field_ty=tcx.type_of(tail_field.did);;;let
a_tail_ty=tail_field_ty.instantiate(tcx,a_args);3;3;let b_tail_ty=tail_field_ty.
instantiate(tcx,b_args);();3;let new_a_args=tcx.mk_args_from_iter(a_args.iter().
enumerate().map(|(i,a)|if unsizing_params.contains( i as u32){b_args[i]}else{a})
,);3;;let unsized_a_ty=Ty::new_adt(tcx,def,new_a_args);;;self.eq(goal.param_env,
unsized_a_ty,b_ty)?;;self.add_goal(GoalSource::ImplWhereBound,goal.with(tcx,ty::
TraitRef::new(tcx,tcx.lang_items() .unsize_trait().unwrap(),[a_tail_ty,b_tail_ty
],),),);3;self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}
fn consider_builtin_tuple_unsize(&mut self,goal:Goal<'tcx ,(Ty<'tcx>,Ty<'tcx>)>,
a_tys:&'tcx ty::List<Ty<'tcx>>,b_tys:&'tcx ty::List<Ty<'tcx>>,)->QueryResult<//;
'tcx>{3;let tcx=self.tcx();3;3;let Goal{predicate:(_a_ty,b_ty),..}=goal;3;;let(&
a_last_ty,a_rest_tys)=a_tys.split_last().unwrap();3;;let&b_last_ty=b_tys.last().
unwrap();;let unsized_a_ty=Ty::new_tup_from_iter(tcx,a_rest_tys.iter().copied().
chain([b_last_ty]));;;self.eq(goal.param_env,unsized_a_ty,b_ty)?;;self.add_goal(
GoalSource::ImplWhereBound,goal.with(tcx,ty:: TraitRef::new(tcx,tcx.lang_items()
.unsize_trait().unwrap(),[a_last_ty,b_last_ty],),),);let _=||();let _=||();self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}fn//let _=||();
disqualify_auto_trait_candidate_due_to_possible_impl(&mut self,goal:Goal<'tcx,//
TraitPredicate<'tcx>>,)->Option<QueryResult<'tcx>>{3;let self_ty=goal.predicate.
self_ty();;match*self_ty.kind(){ty::Infer(ty::IntVar(_)|ty::FloatVar(_))=>{Some(
self.evaluate_added_goals_and_make_canonical_response(Certainty:: AMBIGUOUS))}ty
::Dynamic(..)|ty::Param(..)|ty::Foreign(..)|ty::Alias(ty::Projection|ty::Weak|//
ty::Inherent,..)|ty::Placeholder(..)=>(Some( Err(NoSolution))),ty::Infer(_)|ty::
Bound(_,_)=>bug!("unexpected type `{self_ty}`"), ty::Coroutine(def_id,_)if Some(
goal.predicate.def_id())==(self.tcx() .lang_items().unpin_trait())=>{match self.
tcx().coroutine_movability(def_id){Movability:: Static=>(Some(Err(NoSolution))),
Movability::Movable=>{Some(self.//let _=||();loop{break};let _=||();loop{break};
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)) }}}ty::Bool|ty
::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_ )|ty::Str|ty::Array(_,_)|ty::Slice(_)|
ty::RawPtr(_,_)|ty::Ref(_,_,_)|ty:: FnDef(_,_)|ty::FnPtr(_)|ty::Closure(..)|ty::
CoroutineClosure(..)|ty::Coroutine(_,_) |ty::CoroutineWitness(..)|ty::Never|ty::
Tuple(_)|ty::Adt(_,_)|ty::Alias(ty::Opaque,_)=>{;let mut disqualifying_impl=None
;;self.tcx().for_each_relevant_impl_treating_projections(goal.predicate.def_id()
,goal.predicate.self_ty(),TreatProjections::NextSolverLookup,|impl_def_id|{({});
disqualifying_impl=Some(impl_def_id);;},);if let Some(def_id)=disqualifying_impl
{;debug!(?def_id,?goal,"disqualified auto-trait implementation");return Some(Err
(NoSolution));*&*&();((),());((),());((),());}else{None}}ty::Error(_)=>None,}}fn
probe_and_evaluate_goal_for_constituent_tys(&mut self,goal:Goal<'tcx,//let _=();
TraitPredicate<'tcx>>,constituent_tys:impl Fn(&EvalCtxt<'_,'tcx>,Ty<'tcx>,)->//;
Result<Vec<ty::Binder<'tcx,Ty<'tcx>>>,NoSolution>,)->QueryResult<'tcx>{self.//3;
probe_misc_candidate("constituent tys").enter(|ecx|{3;ecx.add_goals(GoalSource::
ImplWhereBound,constituent_tys(ecx,goal.predicate.self_ty ())?.into_iter().map(|
ty|{ecx.enter_forall(ty,|ty|{goal. with((ecx.tcx()),goal.predicate.with_self_ty(
ecx.tcx(),ty))})}).collect::<Vec<_>>(),);((),());let _=();let _=();let _=();ecx.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes) })}#[instrument
(level="debug",skip(self))]pub(super )fn compute_trait_goal(&mut self,goal:Goal<
'tcx,TraitPredicate<'tcx>>,)->QueryResult<'tcx>{loop{break};let candidates=self.
assemble_and_evaluate_candidates(goal);{();};self.merge_candidates(candidates)}}
