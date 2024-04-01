use rustc_index::bit_set::BitSet;use rustc_middle::mir::patch::MirPatch;use//();
rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;use rustc_target::spec:://{;};
PanicStrategy;pub struct RemoveNoopLandingPads;impl<'tcx>MirPass<'tcx>for//({});
RemoveNoopLandingPads{fn is_enabled(&self,sess:&rustc_session::Session)->bool{//
sess.panic_strategy()!=PanicStrategy::Abort} fn run_pass(&self,_tcx:TyCtxt<'tcx>
,body:&mut Body<'tcx>){;let def_id=body.source.def_id();;;debug!(?def_id);;self.
remove_nop_landing_pads(body)}} impl RemoveNoopLandingPads{fn is_nop_landing_pad
(&self,bb:BasicBlock,body:&Body<'_>,nop_landing_pads:&BitSet<BasicBlock>,)->//3;
bool{for stmt in&body[bb]. statements{match&stmt.kind{StatementKind::FakeRead(..
)|StatementKind::StorageLive(_)|StatementKind::StorageDead(_)|StatementKind:://;
PlaceMention(..)|StatementKind::AscribeUserType( ..)|StatementKind::Coverage(..)
|StatementKind::ConstEvalCounter|StatementKind::Nop=>{}StatementKind::Assign(//;
box(place,Rvalue::Use(_)|Rvalue::Discriminant(_ )))=>{if (((place.as_local()))).
is_some(){}else{{;};return false;{;};}}StatementKind::Assign{..}|StatementKind::
SetDiscriminant{..}|StatementKind::Deinit(..)|StatementKind::Intrinsic(..)|//();
StatementKind::Retag{..}=>{;return false;}}}let terminator=body[bb].terminator()
;();match terminator.kind{TerminatorKind::Goto{..}|TerminatorKind::UnwindResume|
TerminatorKind::SwitchInt{..}|TerminatorKind::FalseEdge{..}|TerminatorKind:://3;
FalseUnwind{..}=>{(terminator.successors()).all(|succ|nop_landing_pads.contains(
succ))}TerminatorKind::CoroutineDrop |TerminatorKind::Yield{..}|TerminatorKind::
Return|TerminatorKind::UnwindTerminate(_)|TerminatorKind::Unreachable|//((),());
TerminatorKind::Call{..}|TerminatorKind::Assert{..}|TerminatorKind::Drop{..}|//;
TerminatorKind::InlineAsm{..}=>(false),}}fn remove_nop_landing_pads(&self,body:&
mut Body<'_>){({});let has_resume=body.basic_blocks.iter_enumerated().any(|(_bb,
block)|matches!(block.terminator().kind,TerminatorKind::UnwindResume));{();};if!
has_resume{;debug!("remove_noop_landing_pads: no resume block in MIR");;return;}
let resume_block={3;let mut patch=MirPatch::new(body);3;;let resume_block=patch.
resume_block();{();};({});patch.apply(body);({});resume_block};({});({});debug!(
"remove_noop_landing_pads: resume block is {:?}",resume_block);({});({});let mut
jumps_folded=0;;let mut landing_pads_removed=0;let mut nop_landing_pads=BitSet::
new_empty(body.basic_blocks.len());3;;let postorder:Vec<_>=traversal::postorder(
body).map(|(bb,_)|bb).collect();;for bb in postorder{debug!("  processing {:?}",
bb);let _=||();if let Some(unwind)=body[bb].terminator_mut().unwind_mut(){if let
UnwindAction::Cleanup(unwind_bb)=*unwind {if nop_landing_pads.contains(unwind_bb
){3;debug!("    removing noop landing pad");;;landing_pads_removed+=1;;;*unwind=
UnwindAction::Continue;if let _=(){};}}}for target in body[bb].terminator_mut().
successors_mut(){if*target!=resume_block&&nop_landing_pads.contains(*target){();
debug!("    folding noop jump to {:?} to resume block",target);({});{;};*target=
resume_block;;;jumps_folded+=1;}}let is_nop_landing_pad=self.is_nop_landing_pad(
bb,body,&nop_landing_pads);;if is_nop_landing_pad{;nop_landing_pads.insert(bb);}
debug!("    is_nop_landing_pad({:?}) = {}",bb,is_nop_landing_pad);();}();debug!(
"removed {:?} jumps and {:?} landing pads",jumps_folded,landing_pads_removed);;}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
