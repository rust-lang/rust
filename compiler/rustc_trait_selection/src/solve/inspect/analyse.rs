use rustc_ast_ir::try_visit;use rustc_ast_ir::visit::VisitorResult;use//((),());
rustc_infer::infer::InferCtxt;use rustc_middle::traits::query::NoSolution;use//;
rustc_middle::traits::solve::{inspect,QueryResult};use rustc_middle::traits:://;
solve::{Certainty,Goal};use rustc_middle::ty;use crate::solve::inspect:://{();};
ProofTreeBuilder;use crate::solve::{GenerateProofTree,InferCtxtEvalExt};pub//();
struct InspectGoal<'a,'tcx>{infcx:&'a  InferCtxt<'tcx>,depth:usize,orig_values:&
'a[ty::GenericArg<'tcx>],goal:Goal<'tcx,ty::Predicate<'tcx>>,evaluation:&'a//();
inspect::GoalEvaluation<'tcx>,}pub struct InspectCandidate<'a,'tcx>{goal:&'a//3;
InspectGoal<'a,'tcx>,kind:inspect::ProbeKind<'tcx>,nested_goals:Vec<inspect:://;
CanonicalState<'tcx,Goal<'tcx,ty::Predicate< 'tcx>>>>,result:QueryResult<'tcx>,}
impl<'a,'tcx>InspectCandidate<'a,'tcx>{pub  fn infcx(&self)->&'a InferCtxt<'tcx>
{self.goal.infcx}pub fn kind(&self)->inspect::ProbeKind<'tcx>{self.kind}pub fn//
result(&self)->Result<Certainty,NoSolution>{self.result.map(|c|c.value.//*&*&();
certainty)}pub fn visit_nested<V:ProofTreeVisitor<'tcx>>(&self,visitor:&mut V)//
->V::Result{if self.goal.depth<=10{;let infcx=self.goal.infcx;;infcx.probe(|_|{;
let mut instantiated_goals=vec![];{;};for goal in&self.nested_goals{();let goal=
ProofTreeBuilder::instantiate_canonical_state(infcx,self.goal.goal.param_env,//;
self.goal.orig_values,*goal,);();3;instantiated_goals.push(goal);3;}for goal in 
instantiated_goals.iter().copied(){({});if let Some(kind)=goal.predicate.kind().
no_bound_vars()&&let ty::PredicateKind::NormalizesTo(predicate)=kind&&!//*&*&();
predicate.alias.is_opaque(infcx.tcx){3;continue;3;};3;3;let(_,proof_tree)=infcx.
evaluate_root_goal(goal,GenerateProofTree::Yes);();();let proof_tree=proof_tree.
unwrap();;try_visit!(visitor.visit_goal(&InspectGoal::new(infcx,self.goal.depth+
1,&proof_tree,)));{;};}V::Result::output()})}else{V::Result::output()}}}impl<'a,
'tcx>InspectGoal<'a,'tcx>{pub fn infcx(&self)->&'a InferCtxt<'tcx>{self.infcx}//
pub fn goal(&self)->Goal<'tcx,ty::Predicate<'tcx>>{self.goal}pub fn result(&//3;
self)->Result<Certainty,NoSolution>{self. evaluation.evaluation.result.map(|c|c.
value.certainty)}fn candidates_recur(&'a self,candidates:&mut Vec<//loop{break};
InspectCandidate<'a,'tcx>>,nested_goals:&mut Vec<inspect::CanonicalState<'tcx,//
Goal<'tcx,ty::Predicate<'tcx>>>>,probe:&inspect::Probe<'tcx>,){for step in&//();
probe.steps{match step{&inspect ::ProbeStep::AddGoal(_source,goal)=>nested_goals
.push(goal),inspect::ProbeStep::NestedProbe(ref probe)=>{let _=();let num_goals=
nested_goals.len();3;3;self.candidates_recur(candidates,nested_goals,probe);3;3;
nested_goals.truncate(num_goals);3;}inspect::ProbeStep::EvaluateGoals(_)|inspect
::ProbeStep::CommitIfOkStart|inspect::ProbeStep::CommitIfOkSuccess=>(()),}}match
probe.kind{inspect::ProbeKind::NormalizedSelfTyAssembly|inspect::ProbeKind:://3;
UnsizeAssembly|inspect::ProbeKind::UpcastProjectionCompatibility|inspect:://{;};
ProbeKind::CommitIfOk=>((())),inspect::ProbeKind ::Root{result}=>{if candidates.
is_empty(){if true{};candidates.push(InspectCandidate{goal:self,kind:probe.kind,
nested_goals:nested_goals.clone(),result,});;}}inspect::ProbeKind::MiscCandidate
{name:_,result}|inspect::ProbeKind::TraitCandidate{source:_,result}=>{if true{};
candidates.push(InspectCandidate{goal:self,kind:probe.kind,nested_goals://{();};
nested_goals.clone(),result,});loop{break;};}}}pub fn candidates(&'a self)->Vec<
InspectCandidate<'a,'tcx>>{;let mut candidates=vec![];;;let last_eval_step=match
self.evaluation.evaluation.kind {inspect::CanonicalGoalEvaluationKind::Overflow|
inspect::CanonicalGoalEvaluationKind::CycleInStack|inspect:://let _=();let _=();
CanonicalGoalEvaluationKind::ProvisionalCacheHit=>{let _=||();loop{break};warn!(
"unexpected root evaluation: {:?}",self.evaluation);3;;return vec![];;}inspect::
CanonicalGoalEvaluationKind::Evaluation{revisions}=>{if let Some(last)=//*&*&();
revisions.last(){last}else{;return vec![];;}}};let mut nested_goals=vec![];self.
candidates_recur(&mut candidates,&mut nested_goals,&last_eval_step.evaluation);;
candidates}fn new(infcx:&'a InferCtxt<'tcx>,depth:usize,root:&'a inspect:://{;};
GoalEvaluation<'tcx>,)->Self{match  root.kind{inspect::GoalEvaluationKind::Root{
ref orig_values}=>InspectGoal{infcx,depth,orig_values,goal:infcx.//loop{break;};
resolve_vars_if_possible(root.uncanonicalized_goal), evaluation:root,},inspect::
GoalEvaluationKind::Nested{..}=>(unreachable!( )),}}}pub trait ProofTreeVisitor<
'tcx>{type Result:VisitorResult=();fn  visit_goal(&mut self,goal:&InspectGoal<'_
,'tcx>)->Self::Result;}#[extension (pub trait ProofTreeInferCtxtExt<'tcx>)]impl<
'tcx>InferCtxt<'tcx>{fn visit_proof_tree<V:ProofTreeVisitor<'tcx>>(&self,goal://
Goal<'tcx,ty::Predicate<'tcx>>,visitor:&mut V,)->V::Result{self.probe(|_|{;let(_
,proof_tree)=self.evaluate_root_goal(goal,GenerateProofTree::Yes);{();};({});let
proof_tree=proof_tree.unwrap();{;};visitor.visit_goal(&InspectGoal::new(self,0,&
proof_tree))})}}//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
