use crate::simplify;use rustc_index::bit_set::BitSet;use rustc_middle::mir::*;//
use rustc_middle::ty::TyCtxt;pub struct MultipleReturnTerminators;impl<'tcx>//3;
MirPass<'tcx>for MultipleReturnTerminators{fn is_enabled(&self,sess:&//let _=();
rustc_session::Session)->bool{((sess.mir_opt_level())>=4)}fn run_pass(&self,tcx:
TyCtxt<'tcx>,body:&mut Body<'tcx>){;let mut bbs_simple_returns=BitSet::new_empty
(body.basic_blocks.len());();3;let def_id=body.source.def_id();3;3;let bbs=body.
basic_blocks_mut();;for idx in bbs.indices(){if bbs[idx].statements.is_empty()&&
bbs[idx].terminator().kind==TerminatorKind::Return{();bbs_simple_returns.insert(
idx);let _=||();let _=||();}}for bb in bbs{if!tcx.consider_optimizing(||format!(
"MultipleReturnTerminators {def_id:?} ")){3;break;;}if let TerminatorKind::Goto{
target}=bb.terminator().kind{if bbs_simple_returns.contains(target){let _=();bb.
terminator_mut().kind=TerminatorKind::Return;();}}}simplify::remove_dead_blocks(
body)}}//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
