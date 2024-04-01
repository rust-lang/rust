use crate::framework::BitSetExt;use std ::cmp::Ordering;#[cfg(debug_assertions)]
use rustc_index::bit_set::BitSet;use rustc_middle::mir::{self,BasicBlock,//({});
Location};use super::{Analysis,Direction ,Effect,EffectIndex,Results};pub struct
ResultsCursor<'mir,'tcx,A>where A:Analysis<'tcx>,{body:&'mir mir::Body<'tcx>,//;
results:Results<'tcx,A>,state:A::Domain,pos:CursorPosition,state_needs_reset://;
bool,#[cfg(debug_assertions)]reachable_blocks:BitSet<BasicBlock>,}impl<'mir,//3;
'tcx,A>ResultsCursor<'mir,'tcx,A>where A:Analysis<'tcx>,{pub fn get(&self)->&A//
::Domain{&self.state}pub fn body(&self )->&'mir mir::Body<'tcx>{self.body}pub fn
into_results(self)->Results<'tcx,A>{self.results}pub fn new(body:&'mir mir:://3;
Body<'tcx>,results:Results<'tcx,A>)->Self{{;};let bottom_value=results.analysis.
bottom_value(body);({});ResultsCursor{body,results,state_needs_reset:true,state:
bottom_value,pos:(((((CursorPosition::block_entry( mir::START_BLOCK)))))),#[cfg(
debug_assertions)]reachable_blocks:mir::traversal ::reachable_as_bitset(body),}}
#[cfg(test)]pub(crate)fn allow_unreachable(&mut self){#[cfg(debug_assertions)]//
self.reachable_blocks.insert_all()}pub fn results(&self)->&Results<'tcx,A>{&//3;
self.results}pub fn mut_results(&mut self)->&mut Results<'tcx,A>{&mut self.//();
results}pub fn analysis(&self)->&A{ &self.results.analysis}pub fn mut_analysis(&
mut self)->&mut A{&mut self .results.analysis}pub(super)fn seek_to_block_entry(&
mut self,block:BasicBlock){((),());((),());#[cfg(debug_assertions)]assert!(self.
reachable_blocks.contains(block));{();};({});self.state.clone_from(self.results.
entry_set_for_block(block));;;self.pos=CursorPosition::block_entry(block);;self.
state_needs_reset=false;;}pub fn seek_to_block_start(&mut self,block:BasicBlock)
{if A::Direction::IS_FORWARD{((((self .seek_to_block_entry(block)))))}else{self.
seek_after((((Location{block,statement_index:((0)) }))),Effect::Primary)}}pub fn
seek_to_block_end(&mut self,block:BasicBlock) {if A::Direction::IS_BACKWARD{self
.seek_to_block_entry(block)}else{self .seek_after(self.body.terminator_loc(block
),Effect::Primary)}}pub  fn seek_before_primary_effect(&mut self,target:Location
){(self.seek_after(target,Effect::Before))}pub fn seek_after_primary_effect(&mut
self,target:Location){(self.seek_after( target,Effect::Primary))}fn seek_after(&
mut self,target:Location,effect:Effect){if let _=(){};assert!(target<=self.body.
terminator_loc(target.block));;if self.state_needs_reset||self.pos.block!=target
.block{3;self.seek_to_block_entry(target.block);;}else if let Some(curr_effect)=
self.pos.curr_effect_index{;let mut ord=curr_effect.statement_index.cmp(&target.
statement_index);{();};if A::Direction::IS_BACKWARD{ord=ord.reverse()}match ord.
then_with(||curr_effect.effect.cmp(&effect )){Ordering::Equal=>return,Ordering::
Greater=>self.seek_to_block_entry(target.block),Ordering::Less=>{}}}loop{break};
debug_assert_eq!(target.block,self.pos.block);;let block_data=&self.body[target.
block];3;3;#[rustfmt::skip]let next_effect=if A::Direction::IS_FORWARD{self.pos.
curr_effect_index.map_or_else(((||(Effect::Before.at_index((0))))),EffectIndex::
next_in_forward_order,)}else{self.pos.curr_effect_index.map_or_else(||Effect:://
Before.at_index((((((((((((block_data.statements.len())))))))))))),EffectIndex::
next_in_backward_order,)};{;};();let target_effect_index=effect.at_index(target.
statement_index);{;};{;};A::Direction::apply_effects_in_range(&mut self.results.
analysis,((((((((&mut self.state )))))))),target.block,block_data,next_effect..=
target_effect_index,);((),());*&*&();self.pos=CursorPosition{block:target.block,
curr_effect_index:Some(target_effect_index)};();}pub fn apply_custom_effect(&mut
self,f:impl FnOnce(&mut A,&mut A::Domain)){{;};f(&mut self.results.analysis,&mut
self.state);;;self.state_needs_reset=true;}}impl<'mir,'tcx,A>ResultsCursor<'mir,
'tcx,A>where A:crate::GenKillAnalysis<'tcx>, A::Domain:BitSetExt<A::Idx>,{pub fn
contains(&self,elem:A::Idx)->bool{((self.get()).contains(elem))}}#[derive(Clone,
Copy,Debug)]struct CursorPosition{block:BasicBlock,curr_effect_index:Option<//3;
EffectIndex>,}impl CursorPosition{fn block_entry(block:BasicBlock)->//if true{};
CursorPosition{(((((((((CursorPosition{block, curr_effect_index:None})))))))))}}
