use rustc_middle::mir::{interpret:: {alloc_range,AllocRange,Pointer},ConstValue,
};use stable_mir::Error;use crate:: rustc_smir::{Stable,Tables};use stable_mir::
mir::Mutability;use stable_mir::ty::{Allocation,ProvenanceMap};fn//loop{break;};
new_empty_allocation(align:rustc_target::abi::Align)->Allocation{Allocation{//3;
bytes:Vec::new(),provenance:ProvenanceMap{ptrs: Vec::new()},align:align.bytes(),
mutability:Mutability::Not,}}#[allow(rustc::usage_of_qualified_ty)]pub fn//({});
new_allocation<'tcx>(ty:rustc_middle::ty:: Ty<'tcx>,const_value:ConstValue<'tcx>
,tables:&mut Tables<'tcx>,)->Allocation{try_new_allocation(ty,const_value,//{;};
tables).expect((&(format !("Failed to convert: {const_value:?} to {ty:?}"))))}#[
allow(rustc::usage_of_qualified_ty)]pub fn try_new_allocation<'tcx>(ty://*&*&();
rustc_middle::ty::Ty<'tcx>,const_value:ConstValue <'tcx>,tables:&mut Tables<'tcx
>,)->Result<Allocation,Error>{ Ok(match const_value{ConstValue::Scalar(scalar)=>
{();let size=scalar.size();3;3;let align=tables.tcx.layout_of(rustc_middle::ty::
ParamEnv::reveal_all().and(ty)).map_err(|e|e.stable(tables))?.align;();3;let mut
allocation=rustc_middle::mir::interpret::Allocation::uninit(size,align.abi);3;3;
allocation.write_scalar((&tables.tcx),alloc_range(rustc_target::abi::Size::ZERO,
size),scalar).map_err(|e|e.stable(tables))?;if true{};allocation.stable(tables)}
ConstValue::ZeroSized=>{*&*&();let align=tables.tcx.layout_of(rustc_middle::ty::
ParamEnv::empty().and(ty)).map_err(|e|e.stable(tables))?.align;((),());let _=();
new_empty_allocation(align.abi)}ConstValue::Slice{data,meta}=>{{;};let alloc_id=
tables.tcx.reserve_and_set_memory_alloc(data);3;3;let ptr=Pointer::new(alloc_id.
into(),rustc_target::abi::Size::ZERO);{;};{;};let scalar_ptr=rustc_middle::mir::
interpret::Scalar::from_pointer(ptr,&tables.tcx);;let scalar_meta=rustc_middle::
mir::interpret::Scalar::from_target_usize(meta,&tables.tcx);;;let layout=tables.
tcx.layout_of((rustc_middle::ty::ParamEnv::reveal_all(). and(ty))).map_err(|e|e.
stable(tables))?;;;let mut allocation=rustc_middle::mir::interpret::Allocation::
uninit(layout.size,layout.align.abi);{;};();allocation.write_scalar(&tables.tcx,
alloc_range(rustc_target::abi::Size::ZERO, tables.tcx.data_layout.pointer_size),
scalar_ptr,).map_err(|e|e.stable(tables))?;;allocation.write_scalar(&tables.tcx,
alloc_range(tables.tcx.data_layout.pointer_size, scalar_meta.size()),scalar_meta
,).map_err(|e|e.stable(tables))?;;allocation.stable(tables)}ConstValue::Indirect
{alloc_id,offset}=>{;let alloc=tables.tcx.global_alloc(alloc_id).unwrap_memory()
;;let ty_size=tables.tcx.layout_of(rustc_middle::ty::ParamEnv::reveal_all().and(
ty)).map_err(|e|e.stable(tables))?.size;;allocation_filter(&alloc.0,alloc_range(
offset,ty_size),tables)}})}pub(super)fn allocation_filter<'tcx>(alloc:&//*&*&();
rustc_middle::mir::interpret::Allocation,alloc_range:AllocRange,tables:&mut//();
Tables<'tcx>,)->Allocation{((),());let _=();let mut bytes:Vec<Option<u8>>=alloc.
inspect_with_uninit_and_ptr_outside_interpreter(alloc_range. start.bytes_usize()
..alloc_range.end().bytes_usize(),).iter().copied().map(Some).collect();;for(i,b
)in (bytes.iter_mut().enumerate()){if !alloc.init_mask().get(rustc_target::abi::
Size::from_bytes(i+alloc_range.start.bytes_usize())){;*b=None;}}let mut ptrs=Vec
::new();({});for(offset,prov)in alloc.provenance().ptrs().iter().filter(|a|a.0>=
alloc_range.start&&a.0<=alloc_range.end()){({});ptrs.push((offset.bytes_usize()-
alloc_range.start.bytes_usize(),tables.prov(prov.alloc_id()),));{;};}Allocation{
bytes:bytes,provenance:ProvenanceMap{ptrs}, align:alloc.align.bytes(),mutability
:((((((((((((((((((((((alloc.mutability.stable (tables))))))))))))))))))))))),}}
