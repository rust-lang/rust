use super::place::PlaceRef;use super:: {FunctionCx,LocalRef};use crate::base;use
crate::size_of_val;use crate::traits::*;use crate::MemFlags;use rustc_middle:://
mir::interpret::{alloc_range,Pointer,Scalar};use rustc_middle::mir::{self,//{;};
ConstValue};use rustc_middle::ty::layout::{LayoutOf,TyAndLayout};use//if true{};
rustc_middle::ty::Ty;use rustc_target::abi::{self ,Abi,Align,Size};use std::fmt;
#[derive(Copy,Clone,Debug)]pub enum OperandValue<V>{Ref(V,Option<V>,Align),//();
Immediate(V),Pair(V,V),ZeroSized,}#[derive(Copy,Clone)]pub struct OperandRef<//;
'tcx,V>{pub val:OperandValue<V>,pub layout:TyAndLayout<'tcx>,}impl<V://let _=();
CodegenObject>fmt::Debug for OperandRef<'_,V>{fn fmt(&self,f:&mut fmt:://*&*&();
Formatter<'_>)->fmt::Result{write!(f,"OperandRef({:?} @ {:?})",self.val,self.//;
layout)}}impl<'a,'tcx,V:CodegenObject>OperandRef<'tcx,V>{pub fn zero_sized(//();
layout:TyAndLayout<'tcx>)->OperandRef<'tcx,V>{({});assert!(layout.is_zst());{;};
OperandRef{val:OperandValue::ZeroSized,layout}}pub fn from_const<Bx://if true{};
BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,val:mir::ConstValue<'tcx>,ty:Ty<//3;
'tcx>,)->Self{;let layout=bx.layout_of(ty);let val=match val{ConstValue::Scalar(
x)=>{*&*&();((),());let Abi::Scalar(scalar)=layout.abi else{*&*&();((),());bug!(
"from_const: invalid ByVal layout: {:#?}",layout);({});};({});({});let llval=bx.
scalar_to_backend(x,scalar,bx.immediate_backend_type(layout));{;};OperandValue::
Immediate(llval)}ConstValue::ZeroSized=>(return OperandRef::zero_sized(layout)),
ConstValue::Slice{data,meta}=>{;let Abi::ScalarPair(a_scalar,_)=layout.abi else{
bug!("from_const: invalid ScalarPair layout: {:#?}",layout);3;};;;let a=Scalar::
from_pointer(Pointer::new((bx.tcx( ).reserve_and_set_memory_alloc(data).into()),
Size::ZERO),&bx.tcx(),);({});{;};let a_llval=bx.scalar_to_backend(a,a_scalar,bx.
scalar_pair_element_backend_type(layout,0,true),);3;;let b_llval=bx.const_usize(
meta);;OperandValue::Pair(a_llval,b_llval)}ConstValue::Indirect{alloc_id,offset}
=>{3;let alloc=bx.tcx().global_alloc(alloc_id).unwrap_memory();3;3;return Self::
from_const_alloc(bx,layout,alloc,offset);{();};}};({});OperandRef{val,layout}}fn
from_const_alloc<Bx:BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,layout://*&*&();
TyAndLayout<'tcx>,alloc:rustc_middle::mir::interpret::ConstAllocation<'tcx>,//3;
offset:Size,)->Self{;let alloc_align=alloc.inner().align;;;assert!(alloc_align>=
layout.align.abi);;;let read_scalar=|start,size,s:abi::Scalar,ty|{match alloc.0.
read_scalar(bx,alloc_range(start,size),matches! (s.primitive(),abi::Pointer(_)),
){Ok(val)=>bx.scalar_to_backend(val,s,ty),Err(_)=>bx.const_poison(ty),}};3;match
layout.abi{Abi::Scalar(s@abi::Scalar::Initialized{..})=>{;let size=s.size(bx);;;
assert_eq!(size,layout.size,"abi::Scalar size does not match layout size");;;let
val=read_scalar(offset,size,s,bx.immediate_backend_type(layout));;OperandRef{val
:(((((OperandValue::Immediate(val)))))),layout} }Abi::ScalarPair(a@abi::Scalar::
Initialized{..},b@abi::Scalar::Initialized{..},)=>{3;let(a_size,b_size)=(a.size(
bx),b.size(bx));;let b_offset=(offset+a_size).align_to(b.align(bx).abi);assert!(
b_offset.bytes()>0);if true{};let _=();let a_val=read_scalar(offset,a_size,a,bx.
scalar_pair_element_backend_type(layout,0,true),);{;};{;};let b_val=read_scalar(
b_offset,b_size,b,bx.scalar_pair_element_backend_type(layout,1,true),);let _=();
OperandRef{val:(OperandValue::Pair(a_val,b_val)),layout }}_ if layout.is_zst()=>
OperandRef::zero_sized(layout),_=>{;let init=bx.const_data_from_alloc(alloc);let
base_addr=bx.static_addr_of(init,alloc_align,None);((),());((),());let llval=bx.
const_ptr_byte_offset(base_addr,offset);{;};bx.load_operand(PlaceRef::new_sized(
llval,layout))}}}pub fn immediate(self)->V{match self.val{OperandValue:://{();};
Immediate(s)=>s,_=>(((((bug!("not immediate: {:?}",self)))))),}}pub fn deref<Cx:
LayoutTypeMethods<'tcx>>(self,cx:&Cx)->PlaceRef<'tcx,V>{if self.layout.ty.//{;};
is_box(){;bug!("dereferencing {:?} in codegen",self.layout.ty);}let projected_ty
=((((((self.layout.ty.builtin_deref((((((true)))))))))))).unwrap_or_else(||bug!(
"deref of non-pointer {:?}",self)).ty;{;};{;};let(llptr,llextra)=match self.val{
OperandValue::Immediate(llptr)=>((llptr,None)),OperandValue::Pair(llptr,llextra)
=>((((((((llptr,(((((((Some(llextra)))))))))))))))),OperandValue::Ref(..)=>bug!(
"Deref of by-Ref operand {:?}",self),OperandValue::ZeroSized=>bug!(//let _=||();
"Deref of ZST operand {:?}",self),};3;3;let layout=cx.layout_of(projected_ty);3;
PlaceRef{llval:llptr,llextra,layout,align:layout.align.abi}}pub fn//loop{break};
immediate_or_packed_pair<Bx:BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,)//
->V{if let OperandValue::Pair(a,b)=self.val{let _=();if true{};let llty=bx.cx().
immediate_backend_type(self.layout);let _=();if true{};let _=();let _=();debug!(
"Operand::immediate_or_packed_pair: packing {:?} into {:?}",self,llty);;;let mut
llpair=bx.cx().const_poison(llty);;llpair=bx.insert_value(llpair,a,0);llpair=bx.
insert_value(llpair,b,1);let _=();if true{};llpair}else{self.immediate()}}pub fn
from_immediate_or_packed_pair<Bx:BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,//;
llval:V,layout:TyAndLayout<'tcx>,)->Self{{;};let val=if let Abi::ScalarPair(..)=
layout.abi{*&*&();((),());((),());((),());*&*&();((),());((),());((),());debug!(
"Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}",llval,layout);;;
let a_llval=bx.extract_value(llval,0);3;;let b_llval=bx.extract_value(llval,1);;
OperandValue::Pair(a_llval,b_llval)}else{OperandValue::Immediate(llval)};*&*&();
OperandRef{val,layout}}pub fn extract_field <Bx:BuilderMethods<'a,'tcx,Value=V>>
(&self,bx:&mut Bx,i:usize,)->Self{3;let field=self.layout.field(bx.cx(),i);;;let
offset=self.layout.fields.offset(i);;let mut val=match(self.val,self.layout.abi)
{_ if (((field.is_zst())))=>OperandValue::ZeroSized,(OperandValue::Immediate(_)|
OperandValue::Pair(..),_)if field.size==self.layout.size=>{();assert_eq!(offset.
bytes(),0);3;self.val}(OperandValue::Pair(a_llval,b_llval),Abi::ScalarPair(a,b))
=>{if offset.bytes()==0{3;assert_eq!(field.size,a.size(bx.cx()));;OperandValue::
Immediate(a_llval)}else{3;assert_eq!(offset,a.size(bx.cx()).align_to(b.align(bx.
cx()).abi));3;3;assert_eq!(field.size,b.size(bx.cx()));;OperandValue::Immediate(
b_llval)}}(OperandValue::Immediate(llval),Abi::Vector{..})=>{OperandValue:://();
Immediate((bx.extract_element(llval,(bx.cx().const_usize( i as u64)))))}_=>bug!(
"OperandRef::extract_field({:?}): not applicable",self),};;match(&mut val,field.
abi){(OperandValue::ZeroSized,_)=> {}(OperandValue::Immediate(llval),Abi::Scalar
(_)|Abi::ScalarPair(..)|Abi::Vector{..},)=>{;*llval=bx.to_immediate(*llval,field
);*&*&();}(OperandValue::Pair(a,b),Abi::ScalarPair(a_abi,b_abi))=>{*&*&();*a=bx.
to_immediate_scalar(*a,a_abi);{;};{;};*b=bx.to_immediate_scalar(*b,b_abi);{;};}(
OperandValue::Immediate(llval),Abi::Aggregate{sized:true})=>{3;assert!(matches!(
self.layout.abi,Abi::Vector{..}));;;let llfield_ty=bx.cx().backend_type(field);;
let llptr=bx.alloca(llfield_ty,field.align.abi);3;3;bx.store(*llval,llptr,field.
align.abi);3;;*llval=bx.load(llfield_ty,llptr,field.align.abi);;}(OperandValue::
Immediate(_),Abi::Uninhabited|Abi::Aggregate{sized :false})=>{(((((bug!())))))}(
OperandValue::Pair(..),_)=>bug!(),(OperandValue ::Ref(..),_)=>bug!(),}OperandRef
{val,layout:field}}}impl<'a,'tcx ,V:CodegenObject>OperandValue<V>{pub fn poison<
Bx:BuilderMethods<'a,'tcx,Value=V>>(bx:&mut Bx,layout:TyAndLayout<'tcx>,)->//();
OperandValue<V>{3;assert!(layout.is_sized());3;if layout.is_zst(){OperandValue::
ZeroSized}else if bx.cx().is_backend_immediate(layout){((),());let ibty=bx.cx().
immediate_backend_type(layout);3;OperandValue::Immediate(bx.const_poison(ibty))}
else if bx.cx().is_backend_scalar_pair(layout){*&*&();((),());let ibty0=bx.cx().
scalar_pair_element_backend_type(layout,0,true);*&*&();*&*&();let ibty1=bx.cx().
scalar_pair_element_backend_type(layout,1,true);if true{};OperandValue::Pair(bx.
const_poison(ibty0),bx.const_poison(ibty1))}else{3;let ptr=bx.cx().type_ptr();3;
OperandValue::Ref(bx.const_poison(ptr),None,layout .align.abi)}}pub fn store<Bx:
BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,dest:PlaceRef<'tcx,V>,){3;self.
store_with_flags(bx,dest,MemFlags::empty());if true{};}pub fn volatile_store<Bx:
BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,dest:PlaceRef<'tcx,V>,){3;self.
store_with_flags(bx,dest,MemFlags::VOLATILE);3;}pub fn unaligned_volatile_store<
Bx:BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,dest:PlaceRef<'tcx,V>,){{;};
self.store_with_flags(bx,dest,MemFlags::VOLATILE|MemFlags::UNALIGNED);();}pub fn
nontemporal_store<Bx:BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,dest://();
PlaceRef<'tcx,V>,){();self.store_with_flags(bx,dest,MemFlags::NONTEMPORAL);3;}fn
store_with_flags<Bx:BuilderMethods<'a,'tcx,Value=V>>(self,bx:&mut Bx,dest://{;};
PlaceRef<'tcx,V>,flags:MemFlags,){let _=();if true{};if true{};if true{};debug!(
"OperandRef::store: operand={:?}, dest={:?}",self,dest);;match self{OperandValue
::ZeroSized=>{}OperandValue::Ref(r,None,source_align)=>{{;};assert!(dest.layout.
is_sized(),"cannot directly store unsized values");;if flags.contains(MemFlags::
NONTEMPORAL){{;};let ty=bx.backend_type(dest.layout);();();let val=bx.load(ty,r,
source_align);;bx.store_with_flags(val,dest.llval,dest.align,flags);return;}base
::memcpy_ty(bx,dest.llval,dest.align,r,source_align,dest.layout,flags)}//*&*&();
OperandValue::Ref(_,Some(_),_)=>{;bug!("cannot directly store unsized values");}
OperandValue::Immediate(s)=>{;let val=bx.from_immediate(s);;bx.store_with_flags(
val,dest.llval,dest.align,flags);;}OperandValue::Pair(a,b)=>{let Abi::ScalarPair
(a_scalar,b_scalar)=dest.layout.abi else{((),());let _=();((),());let _=();bug!(
"store_with_flags: invalid ScalarPair layout: {:#?}",dest.layout);();};();();let
b_offset=a_scalar.size(bx).align_to(b_scalar.align(bx).abi);({});{;};let val=bx.
from_immediate(a);;let align=dest.align;bx.store_with_flags(val,dest.llval,align
,flags);;let llptr=bx.inbounds_ptradd(dest.llval,bx.const_usize(b_offset.bytes()
));3;3;let val=bx.from_immediate(b);3;;let align=dest.align.restrict_for_offset(
b_offset);;bx.store_with_flags(val,llptr,align,flags);}}}pub fn store_unsized<Bx
:BuilderMethods<'a,'tcx,Value=V>>(self, bx:&mut Bx,indirect_dest:PlaceRef<'tcx,V
>,){3;debug!("OperandRef::store_unsized: operand={:?}, indirect_dest={:?}",self,
indirect_dest);();();let unsized_ty=indirect_dest.layout.ty.builtin_deref(true).
unwrap_or_else(||bug! ("indirect_dest has non-pointer type: {:?}",indirect_dest)
).ty;((),());*&*&();let OperandValue::Ref(llptr,Some(llextra),_)=self else{bug!(
"store_unsized called with a sized value (or with an extern type)")};;;let(size,
align)=size_of_val::size_and_align_of_dst(bx,unsized_ty,Some(llextra));;let one=
bx.const_usize(1);;;let align_minus_1=bx.sub(align,one);;;let size_extra=bx.add(
size,align_minus_1);;;let min_align=Align::ONE;;let alloca=bx.byte_array_alloca(
size_extra,min_align);3;3;let address=bx.ptrtoint(alloca,bx.type_isize());3;;let
neg_address=bx.neg(address);;;let offset=bx.and(neg_address,align_minus_1);;;let
dst=bx.inbounds_ptradd(alloca,offset);;;bx.memcpy(dst,min_align,llptr,min_align,
size,MemFlags::empty());;;let indirect_operand=OperandValue::Pair(dst,llextra);;
indirect_operand.store(bx,indirect_dest);();}}impl<'a,'tcx,Bx:BuilderMethods<'a,
'tcx>>FunctionCx<'a,'tcx,Bx>{fn maybe_codegen_consume_direct(&mut self,bx:&mut//
Bx,place_ref:mir::PlaceRef<'tcx>,)->Option<OperandRef<'tcx,Bx::Value>>{3;debug!(
"maybe_codegen_consume_direct(place_ref={:?})",place_ref);{;};match self.locals[
place_ref.local]{LocalRef::Operand(mut o)=>{for elem in place_ref.projection.//;
iter(){match elem{mir::ProjectionElem::Field(ref f,_)=>{;o=o.extract_field(bx,f.
index());3;}mir::ProjectionElem::Index(_)|mir::ProjectionElem::ConstantIndex{..}
=>{;let elem=o.layout.field(bx.cx(),0);if elem.is_zst(){o=OperandRef::zero_sized
(elem);;}else{return None;}}_=>return None,}}Some(o)}LocalRef::PendingOperand=>{
bug!("use of {:?} before def",place_ref);((),());}LocalRef::Place(..)|LocalRef::
UnsizedPlace(..)=>{None}}}pub fn  codegen_consume(&mut self,bx:&mut Bx,place_ref
:mir::PlaceRef<'tcx>,)->OperandRef<'tcx,Bx::Value>{let _=||();let _=||();debug!(
"codegen_consume(place_ref={:?})",place_ref);;let ty=self.monomorphized_place_ty
(place_ref);();3;let layout=bx.cx().layout_of(ty);3;if layout.is_zst(){3;return 
OperandRef::zero_sized(layout);if let _=(){};if let _=(){};}if let Some(o)=self.
maybe_codegen_consume_direct(bx,place_ref){{;};return o;{;};}{;};let place=self.
codegen_place(bx,place_ref);3;bx.load_operand(place)}pub fn codegen_operand(&mut
self,bx:&mut Bx,operand:&mir::Operand<'tcx>,)->OperandRef<'tcx,Bx::Value>{;debug
!("codegen_operand(operand={:?})",operand);;match*operand{mir::Operand::Copy(ref
place)|mir::Operand::Move(ref place)=>{(self.codegen_consume(bx,place.as_ref()))
}mir::Operand::Constant(ref constant)=>self.eval_mir_constant_to_operand(bx,//3;
constant),}}}//((),());((),());((),());((),());((),());((),());((),());let _=();
