use rustc_index::{Idx,IndexVec};use  rustc_middle::mir::*;use rustc_middle::ty::
TyCtxt;#[derive(PartialEq)]pub enum AddCallGuards{AllCallEdges,//*&*&();((),());
CriticalCallEdges,}pub use self::AddCallGuards::*;impl<'tcx>MirPass<'tcx>for//3;
AddCallGuards{fn run_pass(&self,_tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){();self.
add_call_guards(body);3;}}impl AddCallGuards{pub fn add_call_guards(&self,body:&
mut Body<'_>){;let mut pred_count:IndexVec<_,_>=body.basic_blocks.predecessors()
.iter().map(|ps|ps.len()).collect();();();pred_count[START_BLOCK]+=1;3;3;let mut
new_blocks=Vec::new();3;;let cur_len=body.basic_blocks.len();;for block in body.
basic_blocks_mut(){match block. terminator{Some(Terminator{kind:TerminatorKind::
Call{target:Some(ref mut destination),unwind,..},source_info,})if pred_count[*//
destination]>((((1))))&&(matches!(unwind,UnwindAction::Cleanup(_)|UnwindAction::
Terminate(_))||self==&AllCallEdges)=>{;let call_guard=BasicBlockData{statements:
vec![],is_cleanup:block.is_cleanup, terminator:Some(Terminator{source_info,kind:
TerminatorKind::Goto{target:*destination},}),};;let idx=cur_len+new_blocks.len()
;;;new_blocks.push(call_guard);*destination=BasicBlock::new(idx);}_=>{}}}debug!(
"Broke {} N edges",new_blocks.len());;body.basic_blocks_mut().extend(new_blocks)
;let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};}}
