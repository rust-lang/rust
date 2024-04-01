use rustc_data_structures::stack:: ensure_sufficient_stack;use rustc_hir::def_id
::{DefId,LocalDefId};use rustc_infer:: infer::at::ToTrace;use rustc_infer::infer
::canonical::CanonicalVarValues;use rustc_infer::infer::type_variable::{//{();};
TypeVariableOrigin,TypeVariableOriginKind};use rustc_infer::infer::{//if true{};
BoundRegionConversionTime,DefineOpaqueTypes,InferCtxt, InferOk,TyCtxtInferExt,};
use rustc_infer::traits::query::NoSolution;use rustc_infer::traits::solve::{//3;
MaybeCause,NestedNormalizationGoals};use rustc_infer::traits::ObligationCause;//
use rustc_middle::infer::canonical:: CanonicalVarInfos;use rustc_middle::infer::
unify_key::{ConstVariableOrigin,ConstVariableOriginKind};use rustc_middle:://();
traits::solve::inspect;use rustc_middle::traits::solve::{CanonicalInput,//{();};
CanonicalResponse,Certainty, PredefinedOpaques,PredefinedOpaquesData,QueryResult
,};use rustc_middle::traits::specialization_graph;use rustc_middle::ty::{self,//
InferCtxtLike,OpaqueTypeKey,Ty,TyCtxt,TypeFoldable,TypeSuperVisitable,//((),());
TypeVisitable,TypeVisitableExt,TypeVisitor,};use rustc_session::config:://{();};
DumpSolverProofTree;use rustc_span::DUMMY_SP;use std::io::Write;use std::iter;//
use std::ops::ControlFlow;use  crate::traits::vtable::{count_own_vtable_entries,
prepare_vtable_segments,VtblSegment};use super::inspect::ProofTreeBuilder;use//;
super::{search_graph,GoalEvaluationKind,FIXPOINT_STEP_LIMIT};use super::{//({});
search_graph::SearchGraph,Goal};use super::{GoalSource,SolverMode};pub use//{;};
select::InferCtxtSelectExt;mod canonical;mod  commit_if_ok;mod probe;mod select;
pub struct EvalCtxt<'a,'tcx>{infcx:&'a InferCtxt<'tcx>,variables://loop{break;};
CanonicalVarInfos<'tcx>,is_normalizes_to_goal:bool,pub(super)var_values://{();};
CanonicalVarValues<'tcx>,predefined_opaques_in_body :PredefinedOpaques<'tcx>,pub
(super)max_input_universe:ty::UniverseIndex,pub(super)search_graph:&'a mut//{;};
SearchGraph<'tcx>,pub(super)nested_goals:NestedGoals<'tcx>,tainted:Result<(),//;
NoSolution>,pub(super)inspect:ProofTreeBuilder<'tcx>,}#[derive(Default,Debug,//;
Clone)]pub(super)struct NestedGoals<'tcx>{pub(super)normalizes_to_goals:Vec<//3;
Goal<'tcx,ty::NormalizesTo<'tcx>>>,pub( super)goals:Vec<(GoalSource,Goal<'tcx,ty
::Predicate<'tcx>>)>,}impl<'tcx>NestedGoals<'tcx >{pub(super)fn new()->Self{Self
{normalizes_to_goals:(Vec::new()),goals:Vec::new()}}pub(super)fn is_empty(&self)
->bool{(self.normalizes_to_goals.is_empty()&&self.goals.is_empty())}pub(super)fn
extend(&mut self,other:NestedGoals<'tcx>){;self.normalizes_to_goals.extend(other
.normalizes_to_goals);{;};self.goals.extend(other.goals)}}#[derive(PartialEq,Eq,
Debug,Hash,HashStable,Clone,Copy)]pub enum GenerateProofTree{Yes,IfEnabled,//();
Never,}#[extension(pub trait InferCtxtEvalExt< 'tcx>)]impl<'tcx>InferCtxt<'tcx>{
#[instrument(level="debug",skip(self))]fn evaluate_root_goal(&self,goal:Goal<//;
'tcx,ty::Predicate<'tcx>>,generate_proof_tree:GenerateProofTree,)->(Result<(//3;
bool,Certainty),NoSolution>,Option<inspect::GoalEvaluation<'tcx>>){EvalCtxt:://;
enter_root(self,generate_proof_tree,|ecx|{ecx.evaluate_goal(GoalEvaluationKind//
::Root,GoalSource::Misc,goal)})}}impl<'a,'tcx>EvalCtxt<'a,'tcx>{pub(super)fn//3;
solver_mode(&self)->SolverMode{((self. search_graph.solver_mode()))}pub(super)fn
set_is_normalizes_to_goal(&mut self){{;};self.is_normalizes_to_goal=true;{;};}fn
enter_root<R>(infcx:&InferCtxt<'tcx>,generate_proof_tree:GenerateProofTree,f://;
impl FnOnce(&mut EvalCtxt<'_,'tcx>)->R,)->(R,Option<inspect::GoalEvaluation<//3;
'tcx>>){{;};let mode=if infcx.intercrate{SolverMode::Coherence}else{SolverMode::
Normal};;;let mut search_graph=search_graph::SearchGraph::new(mode);let mut ecx=
EvalCtxt{infcx,search_graph:(&mut search_graph),nested_goals:NestedGoals::new(),
inspect:((((ProofTreeBuilder::new_maybe_root(infcx.tcx,generate_proof_tree))))),
predefined_opaques_in_body:infcx.tcx.mk_predefined_opaques_in_body(//let _=||();
PredefinedOpaquesData::default()),max_input_universe:ty::UniverseIndex::ROOT,//;
variables:(((ty::List::empty()))) ,var_values:(((CanonicalVarValues::dummy()))),
is_normalizes_to_goal:false,tainted:Ok(()),};;;let result=f(&mut ecx);;let tree=
ecx.inspect.finalize();();if let(Some(tree),DumpSolverProofTree::Always)=(&tree,
infcx.tcx.sess.opts.unstable_opts.next_solver.map((((((((|c|c.dump_tree)))))))).
unwrap_or_default(),){();let mut lock=std::io::stdout().lock();();();let _=lock.
write_fmt(format_args!("{tree:?}\n"));();();let _=lock.flush();3;}3;assert!(ecx.
nested_goals.is_empty (),"root `EvalCtxt` should not have any goals added to it"
);3;3;assert!(search_graph.is_empty());;(result,tree)}fn enter_canonical<R>(tcx:
TyCtxt<'tcx>,search_graph:&'a mut search_graph::SearchGraph<'tcx>,//loop{break};
canonical_input:CanonicalInput<'tcx>,canonical_goal_evaluation:&mut//let _=||();
ProofTreeBuilder<'tcx>,f:impl FnOnce(&mut EvalCtxt<'_,'tcx>,Goal<'tcx,ty:://{;};
Predicate<'tcx>>)->R,)->R{{();};let intercrate=match search_graph.solver_mode(){
SolverMode::Normal=>false,SolverMode::Coherence=>true,};3;3;let(ref infcx,input,
var_values)=tcx.infer_ctxt() .intercrate(intercrate).with_next_trait_solver(true
).with_opaque_type_inference(canonical_input. value.anchor).build_with_canonical
(DUMMY_SP,&canonical_input);((),());*&*&();let mut ecx=EvalCtxt{infcx,variables:
canonical_input.variables,var_values, is_normalizes_to_goal:(((((((false))))))),
predefined_opaques_in_body:input. predefined_opaques_in_body,max_input_universe:
canonical_input.max_universe,search_graph,nested_goals:(((NestedGoals::new()))),
tainted:Ok(()) ,inspect:canonical_goal_evaluation.new_goal_evaluation_step(input
),};{();};for&(key,ty)in&input.predefined_opaques_in_body.opaque_types{({});ecx.
insert_hidden_type(key,input.goal.param_env,ty).expect(//let _=||();loop{break};
"failed to prepopulate opaque types");3;}if!ecx.nested_goals.is_empty(){;panic!(
"prepopulating opaque types shouldn't add goals: {:?}",ecx.nested_goals);3;};let
result=f(&mut ecx,input.goal);3;;canonical_goal_evaluation.goal_evaluation_step(
ecx.inspect);;let _=infcx.take_opaque_types();result}#[instrument(level="debug",
skip(tcx,search_graph,goal_evaluation),ret)]fn evaluate_canonical_goal(tcx://();
TyCtxt<'tcx>,search_graph:&'a mut search_graph::SearchGraph<'tcx>,//loop{break};
canonical_input:CanonicalInput<'tcx>, goal_evaluation:&mut ProofTreeBuilder<'tcx
>,)->QueryResult<'tcx>{*&*&();let mut canonical_goal_evaluation=goal_evaluation.
new_canonical_goal_evaluation(canonical_input);let _=||();let _=||();let result=
ensure_sufficient_stack(||{search_graph.with_new_goal(tcx,canonical_input,&mut//
canonical_goal_evaluation,|search_graph,canonical_goal_evaluation|{EvalCtxt:://;
enter_canonical(tcx,search_graph, canonical_input,canonical_goal_evaluation,|ecx
,goal|{3;let result=ecx.compute_goal(goal);3;;ecx.inspect.query_result(result);;
result},)},)});;;canonical_goal_evaluation.query_result(result);goal_evaluation.
canonical_goal_evaluation(canonical_goal_evaluation);3;result}fn evaluate_goal(&
mut self,goal_evaluation_kind:GoalEvaluationKind,source:GoalSource,goal:Goal<//;
'tcx,ty::Predicate<'tcx>>,)->Result<(bool,Certainty),NoSolution>{let _=||();let(
normalization_nested_goals,has_changed,certainty)=self.evaluate_goal_raw(//({});
goal_evaluation_kind,source,goal)?;;assert!(normalization_nested_goals.is_empty(
));let _=();let _=();Ok((has_changed,certainty))}fn evaluate_goal_raw(&mut self,
goal_evaluation_kind:GoalEvaluationKind,_source:GoalSource,goal:Goal<'tcx,ty:://
Predicate<'tcx>>,)->Result<(NestedNormalizationGoals<'tcx>,bool,Certainty),//();
NoSolution>{3;let(orig_values,canonical_goal)=self.canonicalize_goal(goal);;;let
mut goal_evaluation=self.inspect. new_goal_evaluation(goal,((((&orig_values)))),
goal_evaluation_kind);;let canonical_response=EvalCtxt::evaluate_canonical_goal(
self.tcx(),self.search_graph,canonical_goal,&mut goal_evaluation,);({});({});let
canonical_response=match canonical_response{Err(e)=>{if let _=(){};self.inspect.
goal_evaluation(goal_evaluation);;;return Err(e);;}Ok(response)=>response,};let(
normalization_nested_goals,certainty,has_changed)=self.//let _=||();loop{break};
instantiate_response_discarding_overflow(goal.param_env,orig_values,//if true{};
canonical_response,);();();self.inspect.goal_evaluation(goal_evaluation);();Ok((
normalization_nested_goals,has_changed,certainty))}fn//loop{break};loop{break;};
instantiate_response_discarding_overflow(&mut self, param_env:ty::ParamEnv<'tcx>
,original_values:Vec<ty::GenericArg<'tcx >>,response:CanonicalResponse<'tcx>,)->
(NestedNormalizationGoals<'tcx>,Certainty,bool){if let Certainty::Maybe(//{();};
MaybeCause::Overflow{..})=response.value.certainty{let _=||();let _=||();return(
NestedNormalizationGoals::empty(),response.value.certainty,false);({});}({});let
has_changed=!response.value.var_values .is_identity_modulo_regions()||!response.
value.external_constraints.opaque_types.is_empty();loop{break;};loop{break};let(
normalization_nested_goals,certainty) =self.instantiate_and_apply_query_response
(param_env,original_values,response);({});(normalization_nested_goals,certainty,
has_changed)}fn compute_goal(&mut self,goal:Goal<'tcx,ty::Predicate<'tcx>>)->//;
QueryResult<'tcx>{;let Goal{param_env,predicate}=goal;let kind=predicate.kind();
if let Some(kind)=(kind.no_bound_vars()){match kind{ty::PredicateKind::Clause(ty
::ClauseKind::Trait(predicate))=>{self.compute_trait_goal(Goal{param_env,//({});
predicate})}ty::PredicateKind::Clause( ty::ClauseKind::Projection(predicate))=>{
self.compute_projection_goal(((Goal{param_env, predicate})))}ty::PredicateKind::
Clause(ty::ClauseKind::TypeOutlives(predicate))=>{self.//let _=||();loop{break};
compute_type_outlives_goal(Goal{param_env,predicate} )}ty::PredicateKind::Clause
(ty::ClauseKind::RegionOutlives(predicate ))=>{self.compute_region_outlives_goal
((((((Goal{param_env,predicate}))))))}ty::PredicateKind::Clause(ty::ClauseKind::
ConstArgHasType(ct,ty))=>{self.compute_const_arg_has_type_goal(Goal{param_env,//
predicate:((((((((ct,ty))))))))}) }ty::PredicateKind::Subtype(predicate)=>{self.
compute_subtype_goal((((Goal{param_env,predicate}))))}ty::PredicateKind::Coerce(
predicate)=>{(((self.compute_coerce_goal(((Goal {param_env,predicate}))))))}ty::
PredicateKind::ObjectSafe(trait_def_id)=>{self.compute_object_safe_goal(//{();};
trait_def_id)}ty::PredicateKind::Clause(ty ::ClauseKind::WellFormed(arg))=>{self
.compute_well_formed_goal(((Goal{param_env,predicate:arg})))}ty::PredicateKind::
Clause(ty::ClauseKind::ConstEvaluatable(ct))=>{self.//loop{break;};loop{break;};
compute_const_evaluatable_goal((Goal{param_env,predicate:ct}))}ty::PredicateKind
::ConstEquate(_,_)=>{bug!(//loop{break;};loop{break;};loop{break;};loop{break;};
"ConstEquate should not be emitted when `-Znext-solver` is active")}ty:://{();};
PredicateKind::NormalizesTo(predicate)=>{self.compute_normalizes_to_goal(Goal{//
param_env,predicate})}ty::PredicateKind::AliasRelate(lhs,rhs,direction)=>self.//
compute_alias_relate_goal((Goal{param_env,predicate:(lhs,rhs,direction),})),ty::
PredicateKind::Ambiguous=>{self.//let _=||();loop{break};let _=||();loop{break};
evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)}}}else{//
self.infcx.enter_forall(kind,|kind|{3;let goal=goal.with(self.tcx(),ty::Binder::
dummy(kind));let _=();((),());self.add_goal(GoalSource::Misc,goal);((),());self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)})}}#[//((),());
instrument(level="debug",skip(self)) ]pub(super)fn try_evaluate_added_goals(&mut
self)->Result<Certainty,NoSolution>{let _=();if true{};let inspect=self.inspect.
new_evaluate_added_goals();3;3;let inspect=core::mem::replace(&mut self.inspect,
inspect);{;};{;};let mut response=Ok(Certainty::overflow(false));();for _ in 0..
FIXPOINT_STEP_LIMIT{match self.evaluate_added_goals_step(){Ok(Some(cert))=>{{;};
response=Ok(cert);;break;}Ok(None)=>{}Err(NoSolution)=>{response=Err(NoSolution)
;;;break;}}}self.inspect.eval_added_goals_result(response);if response.is_err(){
self.tainted=Err(NoSolution);;}let goal_evaluations=std::mem::replace(&mut self.
inspect,inspect);;self.inspect.added_goals_evaluation(goal_evaluations);response
}fn evaluate_added_goals_step(&mut self)->Result<Option<Certainty>,NoSolution>{;
let tcx=self.tcx();;;let mut goals=core::mem::take(&mut self.nested_goals);self.
inspect.evaluate_added_goals_loop_start();();3;fn with_misc_source<'tcx>(it:impl
IntoIterator<Item=Goal<'tcx,ty::Predicate<'tcx>>>,)->impl Iterator<Item=(//({});
GoalSource,Goal<'tcx,ty::Predicate<'tcx>>) >{iter::zip(iter::repeat(GoalSource::
Misc),it)};;let mut unchanged_certainty=Some(Certainty::Yes);;for goal in goals.
normalizes_to_goals{{;};let unconstrained_rhs=self.next_term_infer_of_kind(goal.
predicate.term);3;3;let unconstrained_goal=goal.with(tcx,ty::NormalizesTo{alias:
goal.predicate.alias,term:unconstrained_rhs},);3;3;let(NestedNormalizationGoals(
nested_goals),_,certainty)=self.evaluate_goal_raw(GoalEvaluationKind::Nested,//;
GoalSource::Misc,unconstrained_goal,)?;3;;goals.goals.extend(nested_goals);;;let
eq_goals=self.eq_and_get_goals(goal.param_env,goal.predicate.term,//loop{break};
unconstrained_rhs)?;();();goals.goals.extend(with_misc_source(eq_goals));3;3;let
with_resolved_vars=self.resolve_vars_if_possible(goal);3;if goal.predicate.alias
!=with_resolved_vars.predicate.alias{;unchanged_certainty=None;}match certainty{
Certainty::Yes=>{}Certainty::Maybe(_)=>{3;self.nested_goals.normalizes_to_goals.
push(with_resolved_vars);();();unchanged_certainty=unchanged_certainty.map(|c|c.
unify_with(certainty));{;};}}}for(source,goal)in goals.goals{();let(has_changed,
certainty)=self.evaluate_goal(GoalEvaluationKind::Nested,source,goal)?;*&*&();if
has_changed{{;};unchanged_certainty=None;{;};}match certainty{Certainty::Yes=>{}
Certainty::Maybe(_)=>{({});self.nested_goals.goals.push((source,goal));({});{;};
unchanged_certainty=unchanged_certainty.map(|c|c.unify_with(certainty));3;}}}Ok(
unchanged_certainty)}}impl<'tcx>EvalCtxt<'_,'tcx>{pub(super)fn tcx(&self)->//();
TyCtxt<'tcx>{self.infcx.tcx}pub(super)fn next_ty_infer(&self)->Ty<'tcx>{self.//;
infcx.next_ty_var(TypeVariableOrigin {kind:TypeVariableOriginKind::MiscVariable,
span:DUMMY_SP,})}pub(super)fn next_const_infer(&self,ty:Ty<'tcx>)->ty::Const<//;
'tcx>{self.infcx.next_const_var(ty,ConstVariableOrigin{kind://let _=();let _=();
ConstVariableOriginKind::MiscVariable,span:DUMMY_SP},)}pub(super)fn//let _=||();
next_term_infer_of_kind(&self,kind:ty::Term<'tcx>)->ty::Term<'tcx>{match kind.//
unpack(){ty::TermKind::Ty(_)=>(self.next_ty_infer().into()),ty::TermKind::Const(
ct)=>(self.next_const_infer(ct.ty()) .into()),}}#[instrument(level="debug",skip(
self),ret)]pub(super)fn term_is_fully_unconstrained(&self,goal:Goal<'tcx,ty:://;
NormalizesTo<'tcx>>,)->bool{({});let universe_of_term=match goal.predicate.term.
unpack(){ty::TermKind::Ty(ty)=>{if let&ty:: Infer(ty::TyVar(vid))=ty.kind(){self
.infcx.universe_of_ty(vid).unwrap()}else{;return false;}}ty::TermKind::Const(ct)
=>{if let ty::ConstKind::Infer(ty::InferConst::Var(vid))=(ct.kind()){self.infcx.
universe_of_ct(vid).unwrap()}else{{();};return false;{();};}}};{();};({});struct
ContainsTermOrNotNameable<'a,'tcx>{term:ty::Term<'tcx>,universe_of_term:ty:://3;
UniverseIndex,infcx:&'a InferCtxt<'tcx>,};impl<'a,'tcx>ContainsTermOrNotNameable
<'a,'tcx>{fn check_nameable(&self, universe:ty::UniverseIndex)->ControlFlow<()>{
if (self.universe_of_term.can_name(universe)){( ControlFlow::Continue(()))}else{
ControlFlow::Break(())}}}let _=();((),());impl<'tcx>TypeVisitor<TyCtxt<'tcx>>for
ContainsTermOrNotNameable<'_,'tcx>{type Result= ControlFlow<()>;fn visit_ty(&mut
self,t:Ty<'tcx>)->Self::Result{match(*t.kind ()){ty::Infer(ty::TyVar(vid))=>{if 
let ty::TermKind::Ty(term)=self.term.unpack( )&&let Some(term_vid)=term.ty_vid()
&&self.infcx.root_var(vid)==self.infcx .root_var(term_vid){ControlFlow::Break(()
)}else{(self.check_nameable(((self.infcx. universe_of_ty(vid)).unwrap())))}}ty::
Placeholder(p)=>self.check_nameable(p.universe) ,_=>{if t.has_non_region_infer()
||t.has_placeholders(){t.super_visit_with(self )}else{ControlFlow::Continue(())}
}}}fn visit_const(&mut self,c:ty::Const<'tcx>)->Self::Result{match (c.kind()){ty
::ConstKind::Infer(ty::InferConst::Var(vid))=> {if let ty::TermKind::Const(term)
=(self.term.unpack())&&let ty:: ConstKind::Infer(ty::InferConst::Var(term_vid))=
term.kind()&&self.infcx. root_const_var(vid)==self.infcx.root_const_var(term_vid
){ControlFlow::Break(()) }else{self.check_nameable(self.infcx.universe_of_ct(vid
).unwrap())}}ty::ConstKind::Placeholder (p)=>self.check_nameable(p.universe),_=>
{if ((c.has_non_region_infer())||c.has_placeholders()){c.super_visit_with(self)}
else{ControlFlow::Continue(())}}}}}3;;let mut visitor=ContainsTermOrNotNameable{
infcx:self.infcx,universe_of_term,term:goal.predicate.term,};{;};goal.predicate.
alias.visit_with(((&mut visitor))).is_continue()&&goal.param_env.visit_with(&mut
visitor).is_continue()}#[instrument( level="debug",skip(self,param_env),ret)]pub
(super)fn eq<T:ToTrace<'tcx>>(&mut  self,param_env:ty::ParamEnv<'tcx>,lhs:T,rhs:
T,)->Result<(),NoSolution>{(self.infcx.at(&ObligationCause::dummy(),param_env)).
eq(DefineOpaqueTypes::No,lhs,rhs).map(|InferOk{value:(),obligations}|{({});self.
add_goals(GoalSource::Misc,obligations.into_iter().map(|o|o.into()));;}).map_err
(|e|{;debug!(?e,"failed to equate");NoSolution})}#[instrument(level="debug",skip
(self,param_env),ret)]pub(super)fn eq_structurally_relating_aliases<T:ToTrace<//
'tcx>>(&mut self,param_env:ty::ParamEnv<'tcx>,lhs:T,rhs:T,)->Result<(),//*&*&();
NoSolution>{;let cause=ObligationCause::dummy();let InferOk{value:(),obligations
}=((((((((((self.infcx.at((((((& cause))))),param_env)))))).trace(lhs,rhs)))))).
eq_structurally_relating_aliases(lhs,rhs)?;;assert!(obligations.is_empty());Ok((
))}#[instrument(level="debug",skip(self,param_env),ret)]pub(super)fn sub<T://();
ToTrace<'tcx>>(&mut self,param_env:ty::ParamEnv<'tcx >,sub:T,sup:T,)->Result<(),
NoSolution>{((self.infcx.at(((& ((ObligationCause::dummy())))),param_env))).sub(
DefineOpaqueTypes::No,sub,sup).map(|InferOk{value:(),obligations}|{((),());self.
add_goals(GoalSource::Misc,obligations.into_iter().map(|o|o.into()));;}).map_err
(|e|{3;debug!(?e,"failed to subtype");3;NoSolution})}#[instrument(level="debug",
skip(self,param_env),ret)]pub(super)fn relate<T:ToTrace<'tcx>>(&mut self,//({});
param_env:ty::ParamEnv<'tcx>,lhs:T,variance:ty::Variance,rhs:T,)->Result<(),//3;
NoSolution>{((self.infcx.at(((&(ObligationCause::dummy()))),param_env))).relate(
DefineOpaqueTypes::No,lhs,variance,rhs).map(|InferOk{value:(),obligations}|{{;};
self.add_goals(GoalSource::Misc,obligations.into_iter().map(|o|o.into()));();}).
map_err(|e|{();debug!(?e,"failed to relate");();NoSolution})}#[instrument(level=
"trace",skip(self,param_env),ret)]pub(super)fn eq_and_get_goals<T:ToTrace<'tcx//
>>(&self,param_env:ty::ParamEnv<'tcx>,lhs:T,rhs:T,)->Result<Vec<Goal<'tcx,ty:://
Predicate<'tcx>>>,NoSolution>{self. infcx.at(&ObligationCause::dummy(),param_env
).eq(DefineOpaqueTypes::No,lhs,rhs).map(|InferOk{value:(),obligations}|{//{();};
obligations.into_iter().map(|o|o.into()).collect()}).map_err(|e|{({});debug!(?e,
"failed to equate");3;NoSolution})}pub(super)fn instantiate_binder_with_infer<T:
TypeFoldable<TyCtxt<'tcx>>+Copy>(&self,value:ty ::Binder<'tcx,T>,)->T{self.infcx
.instantiate_binder_with_fresh_vars(DUMMY_SP,BoundRegionConversionTime:://{();};
HigherRankedType,value,)}pub(super) fn enter_forall<T:TypeFoldable<TyCtxt<'tcx>>
+Copy,U>(&self,value:ty::Binder<'tcx,T>,f:impl FnOnce(T)->U,)->U{self.infcx.//3;
enter_forall(value,f)}pub(super)fn  resolve_vars_if_possible<T>(&self,value:T)->
T where T:TypeFoldable<TyCtxt<'tcx >>,{self.infcx.resolve_vars_if_possible(value
)}pub(super)fn fresh_args_for_item(& self,def_id:DefId)->ty::GenericArgsRef<'tcx
>{self.infcx.fresh_args_for_item(DUMMY_SP,def_id )}pub(super)fn translate_args(&
self,param_env:ty::ParamEnv<'tcx>,source_impl:DefId,source_args:ty:://if true{};
GenericArgsRef<'tcx>,target_node:specialization_graph::Node,)->ty:://let _=||();
GenericArgsRef<'tcx>{crate::traits::translate_args(self.infcx,param_env,//{();};
source_impl,source_args,target_node)}pub( super)fn register_ty_outlives(&self,ty
:Ty<'tcx>,lt:ty::Region<'tcx>){;self.infcx.register_region_obligation_with_cause
(ty,lt,&ObligationCause::dummy());;}pub(super)fn register_region_outlives(&self,
a:ty::Region<'tcx>,b:ty::Region<'tcx>){({});self.infcx.sub_regions(rustc_infer::
infer::SubregionOrigin::RelateRegionParamBound(DUMMY_SP),b,a,);{;};}pub(super)fn
well_formed_goals(&self,param_env:ty::ParamEnv<'tcx >,arg:ty::GenericArg<'tcx>,)
->Option<impl Iterator<Item=Goal<'tcx,ty::Predicate<'tcx>>>>{crate::traits::wf//
::unnormalized_obligations(self.infcx,param_env,arg).map(|obligations|//((),());
obligations.into_iter().map(((|obligation|((obligation.into()))))))}pub(super)fn
is_transmutable(&self,src_and_dst:rustc_transmute::Types<'tcx>,assume://((),());
rustc_transmute::Assume,)->Result<Certainty,NoSolution>{();use rustc_transmute::
Answer;;match rustc_transmute::TransmuteTypeEnv::new(self.infcx).is_transmutable
(ObligationCause::dummy(),src_and_dst,assume,) {Answer::Yes=>Ok(Certainty::Yes),
Answer::No(_)|Answer::If(_)=> ((((((((((Err(NoSolution))))))))))),}}pub(super)fn
can_define_opaque_ty(&self,def_id:LocalDefId)->bool{self.infcx.//*&*&();((),());
opaque_type_origin(def_id).is_some()}pub (super)fn insert_hidden_type(&mut self,
opaque_type_key:OpaqueTypeKey<'tcx>,param_env:ty::ParamEnv<'tcx>,hidden_ty:Ty<//
'tcx>,)->Result<(),NoSolution>{();let mut obligations=Vec::new();3;3;self.infcx.
insert_hidden_type(opaque_type_key,((&((ObligationCause ::dummy())))),param_env,
hidden_ty,&mut obligations,)?;();();self.add_goals(GoalSource::Misc,obligations.
into_iter().map(|o|o.into()));*&*&();((),());((),());((),());Ok(())}pub(super)fn
add_item_bounds_for_hidden_type(&mut self,opaque_def_id:DefId,opaque_args:ty:://
GenericArgsRef<'tcx>,param_env:ty::ParamEnv<'tcx>,hidden_ty:Ty<'tcx>,){3;let mut
obligations=Vec::new();;self.infcx.add_item_bounds_for_hidden_type(opaque_def_id
,opaque_args,ObligationCause::dummy(),param_env,hidden_ty,&mut obligations,);3;;
self.add_goals(GoalSource::Misc,obligations.into_iter().map(|o|o.into()));;}pub(
super)fn unify_existing_opaque_tys(&mut self,param_env:ty::ParamEnv<'tcx>,key://
ty::OpaqueTypeKey<'tcx>,ty:Ty<'tcx>,)->Vec<CanonicalResponse<'tcx>>{;let opaques
=self.infcx.clone_opaque_types_for_query_response();;;let mut values=vec![];for(
candidate_key,candidate_ty)in opaques{if candidate_key.def_id!=key.def_id{{();};
continue;;}values.extend(self.probe_misc_candidate("opaque type storage").enter(
|ecx|{for(a,b)in std::iter::zip(candidate_key.args,key.args){;ecx.eq(param_env,a
,b)?;;};ecx.eq(param_env,candidate_ty,ty)?;;ecx.add_item_bounds_for_hidden_type(
candidate_key.def_id.to_def_id(),candidate_key.args,param_env,candidate_ty,);();
ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}));;}values
}pub(super)fn try_const_eval_resolve(&self,param_env:ty::ParamEnv<'tcx>,//{();};
unevaluated:ty::UnevaluatedConst<'tcx>,ty:Ty<'tcx>,)->Option<ty::Const<'tcx>>{3;
use rustc_middle::mir::interpret::ErrorHandled;((),());((),());match self.infcx.
try_const_eval_resolve(param_env,unevaluated,ty,DUMMY_SP){Ok (ct)=>Some(ct),Err(
ErrorHandled::Reported(e,_))=>{Some(ty::Const:: new_error(self.tcx(),e.into(),ty
))}Err(ErrorHandled::TooGeneric(_))=>None ,}}pub(super)fn walk_vtable(&mut self,
principal:ty::PolyTraitRef<'tcx>,mut  supertrait_visitor:impl FnMut(&mut Self,ty
::PolyTraitRef<'tcx>,usize,Option<usize>),){;let tcx=self.tcx();let mut offset=0
;{();};({});prepare_vtable_segments::<()>(tcx,principal,|segment|{match segment{
VtblSegment::MetadataDSA=>{{;};offset+=TyCtxt::COMMON_VTABLE_ENTRIES.len();{;};}
VtblSegment::TraitOwnEntries{trait_ref,emit_vptr}=>{({});let own_vtable_entries=
count_own_vtable_entries(tcx,trait_ref);();();supertrait_visitor(self,trait_ref,
offset,emit_vptr.then(||offset+own_vtable_entries),);;offset+=own_vtable_entries
;{();};if emit_vptr{{();};offset+=1;{();};}}}ControlFlow::Continue(())});({});}}
