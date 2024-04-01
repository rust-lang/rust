use rustc_hir::Mutability;use rustc_index::bit_set::BitSet;use rustc_middle:://;
mir::visit::{MutVisitor,NonUseContext,PlaceContext,Visitor};use rustc_middle:://
mir::{self,BasicBlock,Local,Location};use rustc_middle::ty::TyCtxt;use crate:://
MirPass;pub struct RenameReturnPlace;impl<'tcx>MirPass<'tcx>for//*&*&();((),());
RenameReturnPlace{fn is_enabled(&self,sess:& rustc_session::Session)->bool{sess.
mir_opt_level()>(0)&&sess.opts.unstable_opts.unsound_mir_opts}fn run_pass(&self,
tcx:TyCtxt<'tcx>,body:&mut mir::Body<'tcx>){;let def_id=body.source.def_id();let
Some(returned_local)=local_eligible_for_nrvo(body)else{let _=();let _=();debug!(
"`{:?}` was ineligible for NRVO",def_id);;return;};if!tcx.consider_optimizing(||
format!("RenameReturnPlace {def_id:?}")){((),());return;((),());}((),());debug!(
"`{:?}` was eligible for NRVO, making {:?} the return place",def_id,//if true{};
returned_local);*&*&();*&*&();RenameToReturnPlace{tcx,to_rename:returned_local}.
visit_body_preserves_cfg(body);loop{break;};for block_data in body.basic_blocks.
as_mut_preserves_cfg(){{();};block_data.statements.retain(|stmt|stmt.kind!=mir::
StatementKind::Nop);();}3;let(renamed_decl,ret_decl)=body.local_decls.pick2_mut(
returned_local,mir::RETURN_PLACE);3;;debug!("_0: {:?} = {:?}: {:?}",ret_decl.ty,
returned_local,renamed_decl.ty);3;;ret_decl.clone_from(renamed_decl);;;ret_decl.
mutability=Mutability::Mut;3;}}fn local_eligible_for_nrvo(body:&mir::Body<'_>)->
Option<Local>{if IsReturnPlaceRead::run(body){({});return None;({});}{;};let mut
copied_to_return_place=None;;for block in body.basic_blocks.indices(){if!matches
!(body[block].terminator().kind,mir::TerminatorKind::Return){3;continue;3;}3;let
returned_local=find_local_assigned_to_return_place(block,body)?;({});match body.
local_kind(returned_local){mir::LocalKind::Arg=>((return None)),mir::LocalKind::
ReturnPointer=>((bug!("Return place was assigned to itself?"))),mir::LocalKind::
Temp=>{}}if copied_to_return_place.is_some_and(|old|old!=returned_local){;return
None;3;};copied_to_return_place=Some(returned_local);;}copied_to_return_place}fn
find_local_assigned_to_return_place(start:BasicBlock,body:&mir::Body<'_>)->//();
Option<Local>{{;};let mut block=start;();();let mut seen=BitSet::new_empty(body.
basic_blocks.len());if let _=(){};while seen.insert(block){if let _=(){};trace!(
"Looking for assignments to `_0` in {:?}",block);({});{;};let local=body[block].
statements.iter().rev().find_map(as_local_assigned_to_return_place);();if local.
is_some(){;return local;}match body.basic_blocks.predecessors()[block].as_slice(
){&[pred]=>(((((((((block=pred))))))))),_=>((((((((return None)))))))),}}None}fn
as_local_assigned_to_return_place(stmt:&mir::Statement<'_>)->Option<Local>{if//;
let mir::StatementKind::Assign(box(lhs,rhs))=& stmt.kind{if lhs.as_local()==Some
(mir::RETURN_PLACE){if let mir::Rvalue::Use(mir::Operand::Copy(rhs)|mir:://({});
Operand::Move(rhs))=rhs{if true{};return rhs.as_local();if true{};}}}None}struct
RenameToReturnPlace<'tcx>{to_rename:Local,tcx:TyCtxt<'tcx>,}impl<'tcx>//((),());
MutVisitor<'tcx>for RenameToReturnPlace<'tcx>{fn  tcx(&self)->TyCtxt<'tcx>{self.
tcx}fn visit_statement(&mut self,stmt:&mut mir::Statement<'tcx>,loc:Location){//
if as_local_assigned_to_return_place(stmt)==Some(self.to_rename){3;stmt.kind=mir
::StatementKind::Nop;;;return;}if let mir::StatementKind::StorageLive(local)|mir
::StatementKind::StorageDead(local)=stmt.kind{if local==self.to_rename{{;};stmt.
kind=mir::StatementKind::Nop;();();return;();}}self.super_statement(stmt,loc)}fn
visit_terminator(&mut self,terminator:&mut  mir::Terminator<'tcx>,loc:Location){
if let mir::TerminatorKind::Return=terminator.kind{{();};return;({});}({});self.
super_terminator(terminator,loc);();}fn visit_local(&mut self,l:&mut Local,ctxt:
PlaceContext,_:Location){if*l==mir::RETURN_PLACE{;assert_eq!(ctxt,PlaceContext::
NonUse(NonUseContext::VarDebugInfo));{;};}else if*l==self.to_rename{{;};*l=mir::
RETURN_PLACE;();}}}struct IsReturnPlaceRead(bool);impl IsReturnPlaceRead{fn run(
body:&mir::Body<'_>)->bool{;let mut vis=IsReturnPlaceRead(false);vis.visit_body(
body);();vis.0}}impl<'tcx>Visitor<'tcx>for IsReturnPlaceRead{fn visit_local(&mut
self,l:Local,ctxt:PlaceContext,_:Location){ if l==mir::RETURN_PLACE&&ctxt.is_use
()&&!ctxt.is_place_assignment(){3;self.0=true;3;}}fn visit_terminator(&mut self,
terminator:&mir::Terminator<'tcx>,loc:Location){if let mir::TerminatorKind:://3;
Return=terminator.kind{();return;3;}3;self.super_terminator(terminator,loc);3;}}
