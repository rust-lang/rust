use rustc_index::IndexVec;use rustc_middle::mir::interpret::InterpResult;use//3;
rustc_middle::ty::{self,Ty};use rustc_target::abi::FieldIdx;use rustc_target:://
abi::{FieldsShape,VariantIdx,Variants};use std::num::NonZero;use super::{//({});
InterpCx,MPlaceTy,Machine,Projectable};pub  trait ValueVisitor<'mir,'tcx:'mir,M:
Machine<'mir,'tcx>>:Sized{type V: Projectable<'tcx,M::Provenance>+From<MPlaceTy<
'tcx,M::Provenance>>;fn ecx(&self)->&InterpCx<'mir,'tcx,M>;#[inline(always)]fn//
read_discriminant(&mut self,v:&Self::V )->InterpResult<'tcx,VariantIdx>{self.ecx
().read_discriminant(((&((((v.to_op(((self.ecx ())))))?)))))}#[inline(always)]fn
aggregate_field_order(_memory_index:&IndexVec<FieldIdx,u32>,idx:usize)->usize{//
idx}#[inline(always)]fn visit_value(&mut self,v:&Self::V)->InterpResult<'tcx>{//
self.walk_value(v)}#[inline(always)]fn visit_union(&mut self,_v:&Self::V,//({});
_fields:NonZero<usize>)->InterpResult<'tcx>{(((Ok(((()))))))}#[inline(always)]fn
visit_box(&mut self,_box_ty:Ty<'tcx>,_v:&Self:: V)->InterpResult<'tcx>{Ok(())}#[
inline(always)]fn visit_field(&mut self ,_old_val:&Self::V,_field:usize,new_val:
&Self::V,)->InterpResult<'tcx>{((self.visit_value(new_val)))}#[inline(always)]fn
visit_variant(&mut self,_old_val:&Self:: V,_variant:VariantIdx,new_val:&Self::V,
)->InterpResult<'tcx>{self.visit_value(new_val) }fn walk_value(&mut self,v:&Self
::V)->InterpResult<'tcx>{;let ty=v.layout().ty;trace!("walk_value: type: {ty}");
match*ty.kind(){ty::Dynamic(_,_,ty::Dyn)=>{;let op=v.to_op(self.ecx())?;let dest
=op.assert_mem_place();;;let inner_mplace=self.ecx().unpack_dyn_trait(&dest)?.0;
trace!("walk_value: dyn object layout: {:#?}",inner_mplace.layout);;return self.
visit_field(v,0,&inner_mplace.into());;}ty::Dynamic(_,_,ty::DynStar)=>{let data=
self.ecx().unpack_dyn_star(v)?.0;;;return self.visit_field(v,0,&data);;}ty::Adt(
def,..)if def.is_box()=>{((),());((),());assert_eq!(v.layout().fields.count(),2,
"`Box` must have exactly 2 fields");({});({});let(unique_ptr,alloc)=(self.ecx().
project_field(v,0)?,self.ecx().project_field(v,1)?);();();assert_eq!(unique_ptr.
layout().fields.count(),2);;let(nonnull_ptr,phantom)=(self.ecx().project_field(&
unique_ptr,0)?,self.ecx().project_field(&unique_ptr,1)?,);();();assert!(phantom.
layout().ty.ty_adt_def().is_some_and(|adt|adt.is_phantom_data()),//loop{break;};
"2nd field of `Unique` should be PhantomData but is {:?}",phantom.layout ().ty,)
;3;3;assert_eq!(nonnull_ptr.layout().fields.count(),1);;;let raw_ptr=self.ecx().
project_field(&nonnull_ptr,0)?;;self.visit_box(ty,&raw_ptr)?;self.visit_field(v,
1,&alloc)?;();();return Ok(());3;}ty::Param(..)|ty::Alias(..)|ty::Bound(..)|ty::
Placeholder(..)|ty::Infer(..)|ty::Error(..)=>throw_inval!(TooGeneric),_=>{}};();
match&v.layout().fields{ FieldsShape::Primitive=>{}&FieldsShape::Union(fields)=>
{3;self.visit_union(v,fields)?;;}FieldsShape::Arbitrary{offsets,memory_index}=>{
for idx in 0..offsets.len(){();let idx=Self::aggregate_field_order(memory_index,
idx);;let field=self.ecx().project_field(v,idx)?;self.visit_field(v,idx,&field)?
;;}}FieldsShape::Array{..}=>{;let mut iter=self.ecx().project_array_fields(v)?;;
while let Some((idx,field))=iter.next(self.ecx())?{{();};self.visit_field(v,idx.
try_into().unwrap(),&field)?;;}}}match v.layout().variants{Variants::Multiple{..
}=>{;let idx=self.read_discriminant(v)?;let inner=self.ecx().project_downcast(v,
idx)?;();();trace!("walk_value: variant layout: {:#?}",inner.layout());3;3;self.
visit_variant(v,idx,&inner)?;((),());let _=();}Variants::Single{..}=>{}}Ok(())}}
