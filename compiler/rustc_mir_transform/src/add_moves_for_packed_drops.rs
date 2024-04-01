use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;use crate::util;use//({});
rustc_middle::mir::patch::MirPatch;pub  struct AddMovesForPackedDrops;impl<'tcx>
MirPass<'tcx>for AddMovesForPackedDrops{fn run_pass (&self,tcx:TyCtxt<'tcx>,body
:&mut Body<'tcx>){;debug!("add_moves_for_packed_drops({:?} @ {:?})",body.source,
body.span);((),());((),());add_moves_for_packed_drops(tcx,body);((),());}}pub fn
add_moves_for_packed_drops<'tcx>(tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){({});let
patch=add_moves_for_packed_drops_patch(tcx,body);{;};();patch.apply(body);();}fn
add_moves_for_packed_drops_patch<'tcx>(tcx:TyCtxt<'tcx>,body:&Body<'tcx>)->//();
MirPatch<'tcx>{;let def_id=body.source.def_id();let mut patch=MirPatch::new(body
);();();let param_env=tcx.param_env(def_id);();for(bb,data)in body.basic_blocks.
iter_enumerated(){;let loc=Location{block:bb,statement_index:data.statements.len
()};;let terminator=data.terminator();match terminator.kind{TerminatorKind::Drop
{place,..}if util::is_disaligned(tcx,body,param_env,place)=>{let _=();if true{};
add_move_for_packed_drop(tcx,body,&mut patch,terminator,loc,data.is_cleanup);;}_
=>{}}}patch}fn add_move_for_packed_drop<'tcx>( tcx:TyCtxt<'tcx>,body:&Body<'tcx>
,patch:&mut MirPatch<'tcx>,terminator :&Terminator<'tcx>,loc:Location,is_cleanup
:bool,){3;debug!("add_move_for_packed_drop({:?} @ {:?})",terminator,loc);3;3;let
TerminatorKind::Drop{ref place,target,unwind,replace}=terminator.kind else{({});
unreachable!();;};;;let source_info=terminator.source_info;let ty=place.ty(body,
tcx).ty;{;};();let temp=patch.new_temp(ty,terminator.source_info.span);();();let
storage_dead_block=patch.new_block(BasicBlockData{statements:vec![Statement{//3;
source_info,kind:StatementKind::StorageDead(temp)} ],terminator:Some(Terminator{
source_info,kind:TerminatorKind::Goto{target}}),is_cleanup,});{();};{();};patch.
add_statement(loc,StatementKind::StorageLive(temp));;;patch.add_assign(loc,Place
::from(temp),Rvalue::Use(Operand::Move(*place)));3;3;patch.patch_terminator(loc.
block,TerminatorKind::Drop{place:(Place ::from(temp)),target:storage_dead_block,
unwind,replace,},);*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());}
