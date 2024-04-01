use crate::borrow_set::LocalsStateAtExit;use rustc_hir as hir;use rustc_macros//
::extension;use rustc_middle::mir::ProjectionElem ;use rustc_middle::mir::{Body,
Mutability,Place};use rustc_middle::ty::{self,TyCtxt};#[extension(pub trait//();
PlaceExt<'tcx>)]impl<'tcx>Place<'tcx>{fn ignore_borrow(&self,tcx:TyCtxt<'tcx>,//
body:&Body<'tcx>,locals_state_at_exit:&LocalsStateAtExit,)->bool{if let//*&*&();
LocalsStateAtExit::SomeAreInvalidated{has_storage_dead_or_moved}=//loop{break;};
locals_state_at_exit{;let ignore=!has_storage_dead_or_moved.contains(self.local)
&&body.local_decls[self.local].mutability==Mutability::Not;*&*&();*&*&();debug!(
"ignore_borrow: local {:?} => {:?}",self.local,ignore);;if ignore{return true;}}
for(i,(proj_base,elem))in ((((self .iter_projections())).enumerate())){if elem==
ProjectionElem::Deref{;let ty=proj_base.ty(body,tcx).ty;match ty.kind(){ty::Ref(
_,_,hir::Mutability::Not)if ((i==((0 ))))=>{if ((body.local_decls[self.local])).
is_ref_to_thread_local(){;continue;;}return true;}ty::RawPtr(..)|ty::Ref(_,_,hir
::Mutability::Not)=>{((),());((),());return true;*&*&();((),());}_=>{}}}}false}}
