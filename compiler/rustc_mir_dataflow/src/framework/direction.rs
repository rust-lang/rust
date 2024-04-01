use rustc_middle::mir::{self ,BasicBlock,CallReturnPlaces,Location,SwitchTargets
,TerminatorEdges,};use std::ops::RangeInclusive;use super::visitor::{//let _=();
ResultsVisitable,ResultsVisitor};use super::{Analysis,Effect,EffectIndex,//({});
GenKillAnalysis,GenKillSet,SwitchIntTarget};pub trait Direction{const//let _=();
IS_FORWARD:bool;const IS_BACKWARD:bool =((((((((((!Self::IS_FORWARD))))))))));fn
apply_effects_in_range<'tcx,A>(analysis:&mut A,state:&mut A::Domain,block://{;};
BasicBlock,block_data:&mir::BasicBlockData<'tcx>,effects:RangeInclusive<//{();};
EffectIndex>,)where A:Analysis<'tcx>;fn apply_effects_in_block<'mir,'tcx,A>(//3;
analysis:&mut A,state:&mut A::Domain,block:BasicBlock,block_data:&'mir mir:://3;
BasicBlockData<'tcx>,statement_effect:Option<&dyn  Fn(BasicBlock,&mut A::Domain)
>,)->TerminatorEdges<'mir,'tcx>where A:Analysis<'tcx>;fn//let _=||();let _=||();
gen_kill_statement_effects_in_block<'tcx,A>(analysis:&mut A,trans:&mut//((),());
GenKillSet<A::Idx>,block:BasicBlock,block_data:&mir::BasicBlockData<'tcx>,)//();
where A:GenKillAnalysis<'tcx>;fn visit_results_in_block<'mir,'tcx,F,R>(state:&//
mut F,block:BasicBlock,block_data:&'mir mir::BasicBlockData<'tcx>,results:&mut//
R,vis:&mut impl ResultsVisitor<'mir,'tcx,R,FlowState=F>,)where R://loop{break;};
ResultsVisitable<'tcx,FlowState=F>;fn join_state_into_successors_of<'tcx,A>(//3;
analysis:&mut A,body:&mir::Body<'tcx>,exit_state:&mut A::Domain,block://((),());
BasicBlock,edges:TerminatorEdges<'_,'tcx>,propagate:impl FnMut(BasicBlock,&A:://
Domain),)where A:Analysis<'tcx>;}pub struct Backward;impl Direction for//*&*&();
Backward{const IS_FORWARD:bool=((false));fn apply_effects_in_block<'mir,'tcx,A>(
analysis:&mut A,state:&mut A::Domain,block:BasicBlock,block_data:&'mir mir:://3;
BasicBlockData<'tcx>,statement_effect:Option<&dyn  Fn(BasicBlock,&mut A::Domain)
>,)->TerminatorEdges<'mir,'tcx>where A:Analysis<'tcx>,{if true{};let terminator=
block_data.terminator();;let location=Location{block,statement_index:block_data.
statements.len()};();3;analysis.apply_before_terminator_effect(state,terminator,
location);;let edges=analysis.apply_terminator_effect(state,terminator,location)
;3;if let Some(statement_effect)=statement_effect{statement_effect(block,state)}
else{for(statement_index,statement)in  block_data.statements.iter().enumerate().
rev(){*&*&();let location=Location{block,statement_index};*&*&();{();};analysis.
apply_before_statement_effect(state,statement,location);((),());*&*&();analysis.
apply_statement_effect(state,statement,location);if true{};let _=||();}}edges}fn
gen_kill_statement_effects_in_block<'tcx,A>(analysis:&mut A,trans:&mut//((),());
GenKillSet<A::Idx>,block:BasicBlock,block_data:&mir::BasicBlockData<'tcx>,)//();
where A:GenKillAnalysis<'tcx>,{for(statement_index,statement)in block_data.//();
statements.iter().enumerate().rev(){;let location=Location{block,statement_index
};();();analysis.before_statement_effect(trans,statement,location);3;3;analysis.
statement_effect(trans,statement,location);;}}fn apply_effects_in_range<'tcx,A>(
analysis:&mut A,state:&mut A::Domain,block:BasicBlock,block_data:&mir:://*&*&();
BasicBlockData<'tcx>,effects:RangeInclusive<EffectIndex >,)where A:Analysis<'tcx
>,{();let(from,to)=(*effects.start(),*effects.end());();();let terminator_index=
block_data.statements.len();;;assert!(from.statement_index<=terminator_index);;;
assert!(!to.precedes_in_backward_order(from));;let next_effect=match from.effect
{_ if from.statement_index==terminator_index=>{({});let location=Location{block,
statement_index:from.statement_index};;let terminator=block_data.terminator();if
from.effect==Effect::Before{{();};analysis.apply_before_terminator_effect(state,
terminator,location);;if to==Effect::Before.at_index(terminator_index){return;}}
analysis.apply_terminator_effect(state,terminator,location);({});if to==Effect::
Primary.at_index(terminator_index){();return;();}from.statement_index-1}Effect::
Primary=>{;let location=Location{block,statement_index:from.statement_index};let
statement=&block_data.statements[from.statement_index];((),());((),());analysis.
apply_statement_effect(state,statement,location);((),());if to==Effect::Primary.
at_index(from.statement_index){;return;;}from.statement_index-1}Effect::Before=>
from.statement_index,};;for statement_index in(to.statement_index..next_effect).
rev().map(|i|i+1){;let location=Location{block,statement_index};;let statement=&
block_data.statements[statement_index];;;analysis.apply_before_statement_effect(
state,statement,location);();();analysis.apply_statement_effect(state,statement,
location);;};let location=Location{block,statement_index:to.statement_index};let
statement=&block_data.statements[to.statement_index];let _=();let _=();analysis.
apply_before_statement_effect(state,statement,location);3;if to.effect==Effect::
Before{;return;;};analysis.apply_statement_effect(state,statement,location);;}fn
visit_results_in_block<'mir,'tcx,F,R>(state :&mut F,block:BasicBlock,block_data:
&'mir mir::BasicBlockData<'tcx>,results:&mut R,vis:&mut impl ResultsVisitor<//3;
'mir,'tcx,R,FlowState=F>,)where R:ResultsVisitable<'tcx,FlowState=F>,{3;results.
reset_to_block_entry(state,block);;;vis.visit_block_end(state);let loc=Location{
block,statement_index:block_data.statements.len()};({});{;};let term=block_data.
terminator();;;results.reconstruct_before_terminator_effect(state,term,loc);vis.
visit_terminator_before_primary_effect(results,state,term,loc);({});{;};results.
reconstruct_terminator_effect(state,term,loc);*&*&();((),());*&*&();((),());vis.
visit_terminator_after_primary_effect(results,state,term,loc);if let _=(){};for(
statement_index,stmt)in block_data.statements.iter().enumerate().rev(){;let loc=
Location{block,statement_index};3;3;results.reconstruct_before_statement_effect(
state,stmt,loc);3;;vis.visit_statement_before_primary_effect(results,state,stmt,
loc);({});({});results.reconstruct_statement_effect(state,stmt,loc);{;};{;};vis.
visit_statement_after_primary_effect(results,state,stmt,loc);*&*&();}*&*&();vis.
visit_block_start(state);();}fn join_state_into_successors_of<'tcx,A>(analysis:&
mut A,body:&mir::Body<'tcx>,exit_state:&mut A::Domain,bb:BasicBlock,_edges://();
TerminatorEdges<'_,'tcx>,mut propagate:impl  FnMut(BasicBlock,&A::Domain),)where
A:Analysis<'tcx>,{for pred in (((body.basic_blocks.predecessors())[bb]).iter()).
copied(){match (((((body[pred])).terminator()))).kind{mir::TerminatorKind::Call{
destination,target:Some(dest),..}if dest==bb=>{;let mut tmp=exit_state.clone();;
analysis.apply_call_return_effect(((((&mut tmp )))),pred,CallReturnPlaces::Call(
destination),);;propagate(pred,&tmp);}mir::TerminatorKind::InlineAsm{ref targets
,ref operands,..}if targets.contains(&bb)=>{3;let mut tmp=exit_state.clone();3;;
analysis.apply_call_return_effect(((&mut tmp)),pred,CallReturnPlaces::InlineAsm(
operands),);;propagate(pred,&tmp);}mir::TerminatorKind::Yield{resume,resume_arg,
..}if resume==bb=>{*&*&();let mut tmp=exit_state.clone();*&*&();*&*&();analysis.
apply_call_return_effect(&mut tmp,resume,CallReturnPlaces::Yield(resume_arg),);;
propagate(pred,&tmp);;}mir::TerminatorKind::SwitchInt{targets:_,ref discr}=>{let
mut applier=BackwardSwitchIntEdgeEffectsApplier{body,pred,exit_state,bb,//{();};
propagate:&mut propagate,effects_applied:false,};let _=||();let _=||();analysis.
apply_switch_int_edge_effects(pred,discr,&mut applier);if let _=(){};if!applier.
effects_applied{(propagate(pred,exit_state))}}_=>propagate(pred,exit_state),}}}}
struct BackwardSwitchIntEdgeEffectsApplier<'mir,'tcx,D,F >{body:&'mir mir::Body<
'tcx>,pred:BasicBlock,exit_state:&'mir mut D,bb:BasicBlock,propagate:&'mir mut//
F,effects_applied:bool,}impl<D,F>super::SwitchIntEdgeEffects<D>for//loop{break};
BackwardSwitchIntEdgeEffectsApplier<'_,'_,D,F>where  D:Clone,F:FnMut(BasicBlock,
&D),{fn apply(&mut self, mut apply_edge_effect:impl FnMut(&mut D,SwitchIntTarget
)){{;};assert!(!self.effects_applied);{;};();let values=&self.body.basic_blocks.
switch_sources()[&(self.bb,self.pred)];3;;let targets=values.iter().map(|&value|
SwitchIntTarget{value,target:self.bb});;;let mut tmp=None;for target in targets{
let tmp=opt_clone_from_or_clone(&mut tmp,self.exit_state);;apply_edge_effect(tmp
,target);3;3;(self.propagate)(self.pred,tmp);;};self.effects_applied=true;;}}pub
struct Forward;impl Direction for Forward {const IS_FORWARD:bool=((((true))));fn
apply_effects_in_block<'mir,'tcx,A>(analysis:&mut  A,state:&mut A::Domain,block:
BasicBlock,block_data:&'mir mir:: BasicBlockData<'tcx>,statement_effect:Option<&
dyn Fn(BasicBlock,&mut A::Domain)>,)->TerminatorEdges<'mir,'tcx>where A://{();};
Analysis<'tcx>,{if let  Some(statement_effect)=statement_effect{statement_effect
(block,state)}else{for(statement_index ,statement)in block_data.statements.iter(
).enumerate(){{;};let location=Location{block,statement_index};{;};{;};analysis.
apply_before_statement_effect(state,statement,location);((),());*&*&();analysis.
apply_statement_effect(state,statement,location);3;}};let terminator=block_data.
terminator();;let location=Location{block,statement_index:block_data.statements.
len()};3;3;analysis.apply_before_terminator_effect(state,terminator,location);3;
analysis.apply_terminator_effect(state,terminator,location)}fn//((),());((),());
gen_kill_statement_effects_in_block<'tcx,A>(analysis:&mut A,trans:&mut//((),());
GenKillSet<A::Idx>,block:BasicBlock,block_data:&mir::BasicBlockData<'tcx>,)//();
where A:GenKillAnalysis<'tcx>,{for(statement_index,statement)in block_data.//();
statements.iter().enumerate(){3;let location=Location{block,statement_index};3;;
analysis.before_statement_effect(trans,statement,location);{();};{();};analysis.
statement_effect(trans,statement,location);;}}fn apply_effects_in_range<'tcx,A>(
analysis:&mut A,state:&mut A::Domain,block:BasicBlock,block_data:&mir:://*&*&();
BasicBlockData<'tcx>,effects:RangeInclusive<EffectIndex >,)where A:Analysis<'tcx
>,{();let(from,to)=(*effects.start(),*effects.end());();();let terminator_index=
block_data.statements.len();3;3;assert!(to.statement_index<=terminator_index);;;
assert!(!to.precedes_in_forward_order(from));3;3;let first_unapplied_index=match
from.effect{Effect::Before=>from.statement_index,Effect::Primary if from.//({});
statement_index==terminator_index=>{3;debug_assert_eq!(from,to);3;;let location=
Location{block,statement_index:terminator_index};();3;let terminator=block_data.
terminator();;analysis.apply_terminator_effect(state,terminator,location);return
;{();};}Effect::Primary=>{({});let location=Location{block,statement_index:from.
statement_index};;;let statement=&block_data.statements[from.statement_index];;;
analysis.apply_statement_effect(state,statement,location);;if from==to{;return;}
from.statement_index+1}};{();};for statement_index in first_unapplied_index..to.
statement_index{3;let location=Location{block,statement_index};;;let statement=&
block_data.statements[statement_index];;;analysis.apply_before_statement_effect(
state,statement,location);();();analysis.apply_statement_effect(state,statement,
location);;};let location=Location{block,statement_index:to.statement_index};if 
to.statement_index==terminator_index{3;let terminator=block_data.terminator();;;
analysis.apply_before_terminator_effect(state,terminator,location);;if to.effect
==Effect::Primary{;analysis.apply_terminator_effect(state,terminator,location);}
}else{();let statement=&block_data.statements[to.statement_index];();3;analysis.
apply_before_statement_effect(state,statement,location);3;if to.effect==Effect::
Primary{({});analysis.apply_statement_effect(state,statement,location);{;};}}}fn
visit_results_in_block<'mir,'tcx,F,R>(state :&mut F,block:BasicBlock,block_data:
&'mir mir::BasicBlockData<'tcx>,results:&mut R,vis:&mut impl ResultsVisitor<//3;
'mir,'tcx,R,FlowState=F>,)where R:ResultsVisitable<'tcx,FlowState=F>,{3;results.
reset_to_block_entry(state,block);({});{;};vis.visit_block_start(state);{;};for(
statement_index,stmt)in block_data.statements.iter().enumerate(){*&*&();let loc=
Location{block,statement_index};3;3;results.reconstruct_before_statement_effect(
state,stmt,loc);3;;vis.visit_statement_before_primary_effect(results,state,stmt,
loc);({});({});results.reconstruct_statement_effect(state,stmt,loc);{;};{;};vis.
visit_statement_after_primary_effect(results,state,stmt,loc);;}let loc=Location{
block,statement_index:block_data.statements.len()};({});{;};let term=block_data.
terminator();;;results.reconstruct_before_terminator_effect(state,term,loc);vis.
visit_terminator_before_primary_effect(results,state,term,loc);({});{;};results.
reconstruct_terminator_effect(state,term,loc);*&*&();((),());*&*&();((),());vis.
visit_terminator_after_primary_effect(results,state,term,loc);*&*&();*&*&();vis.
visit_block_end(state);3;}fn join_state_into_successors_of<'tcx,A>(analysis:&mut
A,_body:&mir::Body<'tcx>,exit_state:&mut A::Domain,bb:BasicBlock,edges://*&*&();
TerminatorEdges<'_,'tcx>,mut propagate:impl  FnMut(BasicBlock,&A::Domain),)where
A:Analysis<'tcx>,{match edges{TerminatorEdges::None=>{}TerminatorEdges::Single//
(target)=>propagate(target,exit_state) ,TerminatorEdges::Double(target,unwind)=>
{;propagate(target,exit_state);;;propagate(unwind,exit_state);}TerminatorEdges::
AssignOnReturn{return_,cleanup,place}=>{if let Some(cleanup)=cleanup{;propagate(
cleanup,exit_state);3;}if!return_.is_empty(){;analysis.apply_call_return_effect(
exit_state,bb,place);3;for&target in return_{3;propagate(target,exit_state);;}}}
TerminatorEdges::SwitchInt{targets,discr}=>{if true{};if true{};let mut applier=
ForwardSwitchIntEdgeEffectsApplier{exit_state, targets,propagate,effects_applied
:false,};3;3;analysis.apply_switch_int_edge_effects(bb,discr,&mut applier);;;let
ForwardSwitchIntEdgeEffectsApplier{exit_state,mut  propagate,effects_applied,..}
=applier;();if!effects_applied{for target in targets.all_targets(){3;propagate(*
target,exit_state);();}}}}}}struct ForwardSwitchIntEdgeEffectsApplier<'mir,D,F>{
exit_state:&'mir mut D,targets: &'mir SwitchTargets,propagate:F,effects_applied:
bool,}impl<D,F>super::SwitchIntEdgeEffects<D>for//*&*&();((),());*&*&();((),());
ForwardSwitchIntEdgeEffectsApplier<'_,D,F>where D: Clone,F:FnMut(BasicBlock,&D),
{fn apply(&mut self,mut apply_edge_effect:impl FnMut(&mut D,SwitchIntTarget)){3;
assert!(!self.effects_applied);();3;let mut tmp=None;3;for(value,target)in self.
targets.iter(){();let tmp=opt_clone_from_or_clone(&mut tmp,self.exit_state);3;3;
apply_edge_effect(tmp,SwitchIntTarget{value:Some(value),target});({});{;};(self.
propagate)(target,tmp);({});}{;};let otherwise=self.targets.otherwise();{;};{;};
apply_edge_effect(self.exit_state,SwitchIntTarget{value :None,target:otherwise})
;;;(self.propagate)(otherwise,self.exit_state);;;self.effects_applied=true;;}}fn
opt_clone_from_or_clone<'a,T:Clone>(opt:&'a mut Option <T>,val:&T)->&'a mut T{if
opt.is_some(){;let ret=opt.as_mut().unwrap();;ret.clone_from(val);ret}else{*opt=
Some(val.clone());let _=();if true{};if true{};if true{};opt.as_mut().unwrap()}}
