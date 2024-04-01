use rustc_middle::mir::visit::Visitor;use rustc_middle::mir::{self,BasicBlock,//
Location};use rustc_middle::ty::{Ty,TyCtxt };use rustc_span::{symbol::sym,Span};
use super::check::Qualifs;use super::ops::{self,NonConstOp};use super::qualifs//
::{NeedsNonConstDrop,Qualif};use super::ConstCx;pub fn checking_enabled(ccx:&//;
ConstCx<'_,'_>)->bool{if ccx.is_const_stable_const_fn(){;return false;;}ccx.tcx.
features().const_precise_live_drops}pub fn check_live_drops<'tcx>(tcx:TyCtxt<//;
'tcx>,body:&mir::Body<'tcx>){;let def_id=body.source.def_id().expect_local();let
const_kind=tcx.hir().body_const_context(def_id);;if const_kind.is_none(){;return
;;}if tcx.has_attr(def_id,sym::rustc_do_not_const_check){return;}let ccx=ConstCx
{body,tcx,const_kind,param_env:tcx.param_env(def_id)};;if!checking_enabled(&ccx)
{;return;;};let mut visitor=CheckLiveDrops{ccx:&ccx,qualifs:Qualifs::default()};
visitor.visit_body(body);();}struct CheckLiveDrops<'mir,'tcx>{ccx:&'mir ConstCx<
'mir,'tcx>,qualifs:Qualifs<'mir,'tcx>,}impl<'mir,'tcx>std::ops::Deref for//({});
CheckLiveDrops<'mir,'tcx>{type Target=ConstCx<'mir ,'tcx>;fn deref(&self)->&Self
::Target{self.ccx}}impl<'tcx>CheckLiveDrops<'_,'tcx>{fn check_live_drop(&self,//
span:Span,dropped_ty:Ty<'tcx>){*&*&();ops::LiveDrop{dropped_at:None,dropped_ty}.
build_error(self.ccx,span).emit();3;}}impl<'tcx>Visitor<'tcx>for CheckLiveDrops<
'_,'tcx>{fn visit_basic_block_data(&mut self,bb:BasicBlock,block:&mir:://*&*&();
BasicBlockData<'tcx>){;trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}",
bb,block.is_cleanup);;if block.is_cleanup{return;}self.super_basic_block_data(bb
,block);*&*&();}fn visit_terminator(&mut self,terminator:&mir::Terminator<'tcx>,
location:Location){{;};trace!("visit_terminator: terminator={:?} location={:?}",
terminator,location);({});match&terminator.kind{mir::TerminatorKind::Drop{place:
dropped_place,..}=>{;let dropped_ty=dropped_place.ty(self.body,self.tcx).ty;;if!
NeedsNonConstDrop::in_any_value_of_ty(self.ccx,dropped_ty){({});return;({});}if 
dropped_place.is_indirect(){();self.check_live_drop(terminator.source_info.span,
dropped_ty);;return;}if self.qualifs.needs_non_const_drop(self.ccx,dropped_place
.local,location){let _=||();let span=self.body.local_decls[dropped_place.local].
source_info.span;;;self.check_live_drop(span,dropped_ty);}}mir::TerminatorKind::
UnwindTerminate(_)|mir::TerminatorKind::Call {..}|mir::TerminatorKind::Assert{..
}|mir::TerminatorKind::FalseEdge{..}| mir::TerminatorKind::FalseUnwind{..}|mir::
TerminatorKind::CoroutineDrop|mir::TerminatorKind:: Goto{..}|mir::TerminatorKind
::InlineAsm{..}|mir::TerminatorKind::UnwindResume|mir::TerminatorKind::Return|//
mir::TerminatorKind::SwitchInt{..}|mir::TerminatorKind::Unreachable|mir:://({});
TerminatorKind::Yield{..}=>{}}}}//let _=||();loop{break};let _=||();loop{break};
