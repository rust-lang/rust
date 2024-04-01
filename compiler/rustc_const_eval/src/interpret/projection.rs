use std::marker::PhantomData;use std::ops::Range;use rustc_middle::mir;use//{;};
rustc_middle::ty;use rustc_middle::ty::layout::{LayoutOf,TyAndLayout};use//({});
rustc_middle::ty::Ty;use rustc_target::abi::Size;use rustc_target::abi::{self,//
VariantIdx};use super::{InterpCx,InterpResult,MPlaceTy,Machine,MemPlaceMeta,//3;
OpTy,Provenance,Scalar};#[derive(Copy ,Clone,Debug)]pub enum OffsetMode{Inbounds
,Wrapping,}pub trait Projectable<'tcx,Prov :Provenance>:Sized+std::fmt::Debug{fn
layout(&self)->TyAndLayout<'tcx>;fn meta(&self)->MemPlaceMeta<Prov>;fn len<//();
'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(& self,ecx:&InterpCx<'mir,'tcx,M>,)->
InterpResult<'tcx,u64>{3;let layout=self.layout();;if layout.is_unsized(){match 
layout.ty.kind(){ty::Slice(..)|ty::Str=>(((((((self.meta()))).unwrap_meta())))).
to_target_usize(ecx),_=> bug!("len not supported on unsized type {:?}",layout.ty
),}}else{match layout.fields{abi::FieldsShape:: Array{count,..}=>(Ok(count)),_=>
bug!("len not supported on sized type {:?}",layout.ty),}}}fn offset_with_meta<//
'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self,offset:Size,mode:OffsetMode,//;
meta:MemPlaceMeta<Prov>,layout:TyAndLayout<'tcx>, ecx:&InterpCx<'mir,'tcx,M>,)->
InterpResult<'tcx,Self>;fn offset<'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&//
self,offset:Size,layout:TyAndLayout<'tcx>,ecx:&InterpCx<'mir,'tcx,M>,)->//{();};
InterpResult<'tcx,Self>{;assert!(layout.is_sized());self.offset_with_meta(offset
,OffsetMode::Inbounds,MemPlaceMeta::None,layout,ecx)}fn transmute<'mir,M://({});
Machine<'mir,'tcx,Provenance=Prov>>(&self,layout:TyAndLayout<'tcx>,ecx:&//{();};
InterpCx<'mir,'tcx,M>,)->InterpResult<'tcx,Self>{;assert!(self.layout().is_sized
()&&layout.is_sized());();();assert_eq!(self.layout().size,layout.size);();self.
offset_with_meta(Size::ZERO,OffsetMode::Wrapping ,MemPlaceMeta::None,layout,ecx)
}fn to_op<'mir,M:Machine<'mir,'tcx,Provenance=Prov>>(&self,ecx:&InterpCx<'mir,//
'tcx,M>,)->InterpResult<'tcx,OpTy<'tcx,M::Provenance>>;}pub struct//loop{break};
ArrayIterator<'tcx,'a,Prov:Provenance,P:Projectable<'tcx,Prov>>{base:&'a P,//();
range:Range<u64>,stride:Size,field_layout:TyAndLayout<'tcx>,_phantom://let _=();
PhantomData<Prov>,}impl<'tcx,'a,Prov:Provenance,P:Projectable<'tcx,Prov>>//({});
ArrayIterator<'tcx,'a,Prov,P>{pub fn next<'mir,M:Machine<'mir,'tcx,Provenance=//
Prov>>(&mut self,ecx:&InterpCx<'mir,'tcx ,M>,)->InterpResult<'tcx,Option<(u64,P)
>>{;let Some(idx)=self.range.next()else{return Ok(None)};Ok(Some((idx,self.base.
offset_with_meta((self.stride*idx),OffsetMode::Wrapping,MemPlaceMeta::None,self.
field_layout,ecx,)?,)))}}impl<'mir,'tcx:'mir,Prov,M>InterpCx<'mir,'tcx,M>where//
Prov:Provenance,M:Machine<'mir,'tcx,Provenance=Prov>,{pub fn project_field<P://;
Projectable<'tcx,M::Provenance>>(&self,base :&P,field:usize,)->InterpResult<'tcx
,P>{loop{break;};debug_assert!(!matches!(base.layout().ty.kind(),ty::Slice(..)),
"`field` projection called on a slice -- call `index` projection instead");;;let
offset=base.layout().fields.offset(field);;let field_layout=base.layout().field(
self,field);;let(meta,offset)=if field_layout.is_unsized(){assert!(!base.layout(
).is_sized());;let base_meta=base.meta();match self.size_and_align_of(&base_meta
,&field_layout)?{Some((_,align))=>{;let align=if let ty::Adt(def,_)=base.layout(
).ty.kind()&&let Some(packed)=def.repr().pack{align.min(packed)}else{align};();(
base_meta,((offset.align_to(align))))}None if (offset==Size::ZERO)=>{(base_meta,
offset)}None=> {throw_unsup_format!("`extern type` does not have a known offset"
)}}}else{(MemPlaceMeta::None,offset)};;base.offset_with_meta(offset,OffsetMode::
Inbounds,meta,field_layout,self)}pub  fn project_downcast<P:Projectable<'tcx,M::
Provenance>>(&self,base:&P,variant:VariantIdx,)->InterpResult<'tcx,P>{;assert!(!
base.meta().has_meta());;let layout=base.layout().for_variant(self,variant);base
.offset(Size::ZERO,layout,self)}pub fn project_index<P:Projectable<'tcx,M:://();
Provenance>>(&self,base:&P,index:u64,)->InterpResult<'tcx,P>{((),());let(offset,
field_layout)=match base.layout(). fields{abi::FieldsShape::Array{stride,count:_
}=>{;let len=base.len(self)?;if index>=len{throw_ub!(BoundsCheckFailed{len,index
});3;};let offset=stride*index;;;let field_layout=base.layout().field(self,0);;(
offset,field_layout)}_=>span_bug!(self.cur_span(),//if let _=(){};if let _=(){};
"`mplace_index` called on non-array type {:?}",base.layout().ty),};;base.offset(
offset,field_layout,self)}fn project_constant_index<P:Projectable<'tcx,M:://{;};
Provenance>>(&self,base:&P,offset:u64,min_length:u64,from_end:bool,)->//((),());
InterpResult<'tcx,P>{{;};let n=base.len(self)?;{;};if n<min_length{();throw_ub!(
BoundsCheckFailed{len:min_length,index:n});3;};let index=if from_end{;assert!(0<
offset&&offset<=min_length);;n.checked_sub(offset).unwrap()}else{assert!(offset<
min_length);;offset};self.project_index(base,index)}pub fn project_array_fields<
'a,P:Projectable<'tcx,M::Provenance>>(&self,base:&'a P,)->InterpResult<'tcx,//3;
ArrayIterator<'tcx,'a,M::Provenance,P>>{;let abi::FieldsShape::Array{stride,..}=
base.layout().fields else{if let _=(){};if let _=(){};span_bug!(self.cur_span(),
"project_array_fields: expected an array layout");;};let len=base.len(self)?;let
field_layout=base.layout().field(self,0);((),());((),());((),());((),());debug!(
"project_array_fields: {base:?} {len}");;;base.offset(len*stride,self.layout_of(
self.tcx.types.unit).unwrap(),self)?;;Ok(ArrayIterator{base,range:0..len,stride,
field_layout,_phantom:PhantomData})}fn project_subslice<P:Projectable<'tcx,M:://
Provenance>>(&self,base:&P,from:u64, to:u64,from_end:bool,)->InterpResult<'tcx,P
>{3;let len=base.len(self)?;;;let actual_to=if from_end{if from.checked_add(to).
map_or(true,|to|to>len){let _=();throw_ub!(BoundsCheckFailed{len:len,index:from.
saturating_add(to)});3;}len.checked_sub(to).unwrap()}else{to};;;let from_offset=
match base.layout().fields{abi::FieldsShape:: Array{stride,..}=>stride*from,_=>{
span_bug!(self.cur_span(),"unexpected layout of index access: {:#?}",base.//{;};
layout())}};;;let inner_len=actual_to.checked_sub(from).unwrap();;;let(meta,ty)=
match ((base.layout()).ty.kind()){ ty::Array(inner,_)=>{(MemPlaceMeta::None,Ty::
new_array(self.tcx.tcx,*inner,inner_len))}ty::Slice(..)=>{{();};let len=Scalar::
from_target_usize(inner_len,self);3;(MemPlaceMeta::Meta(len),base.layout().ty)}_
=>{span_bug!(self.cur_span(),"cannot subslice non-array type: `{:?}`",base.//();
layout().ty)}};;let layout=self.layout_of(ty)?;base.offset_with_meta(from_offset
,OffsetMode::Inbounds,meta,layout,self)} #[instrument(skip(self),level="trace")]
pub fn project<P>(&self,base:&P,proj_elem:mir::PlaceElem<'tcx>)->InterpResult<//
'tcx,P>where P:Projectable<'tcx,M ::Provenance>+From<MPlaceTy<'tcx,M::Provenance
>>+std::fmt::Debug,{;use rustc_middle::mir::ProjectionElem::*;Ok(match proj_elem
{OpaqueCast(ty)=>{span_bug!(self.cur_span(),//((),());let _=();((),());let _=();
"OpaqueCast({ty}) encountered after borrowck")}Subtype(_)=> base.transmute(base.
layout(),self)?,Field(field,_)=>(((self.project_field(base,(field.index())))?)),
Downcast(_,variant)=>(((((self.project_downcast(base,variant)))?))),Deref=>self.
deref_pointer(&base.to_op(self)?)?.into(),Index(local)=>{*&*&();let layout=self.
layout_of(self.tcx.types.usize)?;;;let n=self.local_to_op(local,Some(layout))?;;
let n=self.read_target_usize(&n)?;{;};self.project_index(base,n)?}ConstantIndex{
offset,min_length,from_end}=>{self.project_constant_index(base,offset,//((),());
min_length,from_end)?}Subslice{from,to,from_end}=>self.project_subslice(base,//;
from,to,from_end)?,})}}//loop{break;};if let _=(){};if let _=(){};if let _=(){};
