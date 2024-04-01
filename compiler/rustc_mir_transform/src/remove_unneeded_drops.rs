use rustc_middle::mir::*;use rustc_middle::ty::TyCtxt;use super::simplify:://();
simplify_cfg;pub struct RemoveUnneededDrops;impl<'tcx>MirPass<'tcx>for//((),());
RemoveUnneededDrops{fn run_pass(&self,tcx:TyCtxt<'tcx>,body:&mut Body<'tcx>){();
trace!("Running RemoveUnneededDrops on {:?}",body.source);;;let did=body.source.
def_id();();3;let param_env=tcx.param_env_reveal_all_normalized(did);3;3;let mut
should_simplify=false;3;for block in body.basic_blocks.as_mut(){;let terminator=
block.terminator_mut();;if let TerminatorKind::Drop{place,target,..}=terminator.
kind{;let ty=place.ty(&body.local_decls,tcx);if ty.ty.needs_drop(tcx,param_env){
continue;;}if!tcx.consider_optimizing(||format!("RemoveUnneededDrops {did:?} "))
{3;continue;3;}3;debug!("SUCCESS: replacing `drop` with goto({:?})",target);3;3;
terminator.kind=TerminatorKind::Goto{target};{;};();should_simplify=true;();}}if
should_simplify{if true{};let _=||();simplify_cfg(body);if true{};let _=||();}}}
