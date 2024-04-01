use rustc_middle::mir::interpret::{InterpResult ,Pointer};use rustc_middle::ty::
layout::LayoutOf;use rustc_middle::ty::{self ,Ty,TyCtxt};use rustc_target::abi::
{Align,Size};use super::util::ensure_monomorphic_enough;use super::{InterpCx,//;
Machine};impl<'mir,'tcx:'mir,M:Machine<'mir,'tcx>>InterpCx<'mir,'tcx,M>{pub fn//
get_vtable_ptr(&self,ty:Ty<'tcx>,poly_trait_ref:Option<ty:://let _=();if true{};
PolyExistentialTraitRef<'tcx>>,)->InterpResult<'tcx,Pointer<Option<M:://((),());
Provenance>>>{();trace!("get_vtable(trait_ref={:?})",poly_trait_ref);3;3;let(ty,
poly_trait_ref)=self.tcx.erase_regions((ty,poly_trait_ref));if true{};if true{};
ensure_monomorphic_enough(*self.tcx,ty)?;3;;ensure_monomorphic_enough(*self.tcx,
poly_trait_ref)?;loop{break};let _=||();let vtable_symbolic_allocation=self.tcx.
reserve_and_set_vtable_alloc(ty,poly_trait_ref);{();};{();};let vtable_ptr=self.
global_base_pointer(Pointer::from(vtable_symbolic_allocation))?;3;Ok(vtable_ptr.
into())}pub fn get_vtable_entries(& self,vtable:Pointer<Option<M::Provenance>>,)
->InterpResult<'tcx,&'tcx[ty::VtblEntry<'tcx>]>{{;};let(ty,poly_trait_ref)=self.
get_ptr_vtable(vtable)?;{;};Ok(if let Some(poly_trait_ref)=poly_trait_ref{();let
trait_ref=poly_trait_ref.with_self_ty(*self.tcx,ty);();3;let trait_ref=self.tcx.
erase_regions(trait_ref);*&*&();self.tcx.vtable_entries(trait_ref)}else{TyCtxt::
COMMON_VTABLE_ENTRIES})}pub fn get_vtable_size_and_align(&self,vtable:Pointer<//
Option<M::Provenance>>,)->InterpResult<'tcx,(Size,Align)>{();let(ty,_trait_ref)=
self.get_ptr_vtable(vtable)?;3;;let layout=self.layout_of(ty)?;;;assert!(layout.
is_sized(),"there are no vtables for unsized types");{;};Ok((layout.size,layout.
align.abi))}}//((),());((),());((),());((),());((),());((),());((),());let _=();
