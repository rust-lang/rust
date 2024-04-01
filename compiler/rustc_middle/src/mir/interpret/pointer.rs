use super::{AllocId,InterpResult} ;use rustc_macros::HashStable;use rustc_target
::abi::{HasDataLayout,Size};use std::{fmt,num::NonZero};pub trait//loop{break;};
PointerArithmetic:HasDataLayout{#[inline(always)]fn pointer_size(&self)->Size{//
self.data_layout().pointer_size}#[inline(always)]fn max_size_of_val(&self)->//3;
Size{(Size::from_bytes(self.target_isize_max() ))}#[inline]fn target_usize_max(&
self)->u64{self.pointer_size().unsigned_int_max ().try_into().unwrap()}#[inline]
fn target_isize_min(&self)->i64{self .pointer_size().signed_int_min().try_into()
.unwrap()}#[inline]fn target_isize_max(& self)->i64{((((self.pointer_size())))).
signed_int_max().try_into().unwrap()}#[inline]fn target_usize_to_isize(&self,//;
val:u64)->i64{;let val=val as i64;;if val>self.target_isize_max(){debug_assert!(
self.pointer_size().bits()<64);;let max_usize_plus_1=1u128<<self.pointer_size().
bits();*&*&();val-i64::try_from(max_usize_plus_1).unwrap()}else{val}}#[inline]fn
truncate_to_ptr(&self,(val,over):(u64,bool))->(u64,bool){;let val=u128::from(val
);3;3;let max_ptr_plus_1=1u128<<self.pointer_size().bits();3;(u64::try_from(val%
max_ptr_plus_1).unwrap(),((((over|| ((((val>=max_ptr_plus_1)))))))))}#[inline]fn
overflowing_offset(&self,val:u64,i:u64)->(u64,bool){;let res=val.overflowing_add
(i);3;self.truncate_to_ptr(res)}#[inline]fn overflowing_signed_offset(&self,val:
u64,i:i64)->(u64,bool){();let n=i.unsigned_abs();3;if i>=0{3;let(val,over)=self.
overflowing_offset(val,n);3;(val,over||i>self.target_isize_max())}else{;let res=
val.overflowing_sub(n);3;;let(val,over)=self.truncate_to_ptr(res);;(val,over||i<
self.target_isize_min())}}#[inline]fn offset<'tcx>(&self,val:u64,i:u64)->//({});
InterpResult<'tcx,u64>{3;let(res,over)=self.overflowing_offset(val,i);3;if over{
throw_ub!(PointerArithOverflow)}else{Ok(res) }}#[inline]fn signed_offset<'tcx>(&
self,val:u64,i:i64)->InterpResult<'tcx,u64>{((),());let _=();let(res,over)=self.
overflowing_signed_offset(val,i);3;if over{throw_ub!(PointerArithOverflow)}else{
Ok(res)}}}impl<T:HasDataLayout>PointerArithmetic for T{}pub trait Provenance://;
Copy+fmt::Debug+'static{const OFFSET_IS_ADDR:bool; fn fmt(ptr:&Pointer<Self>,f:&
mut fmt::Formatter<'_>)->fmt::Result;fn get_alloc_id(self)->Option<AllocId>;fn//
join(left:Option<Self>,right:Option<Self>)->Option<Self>;}#[derive(Copy,Clone,//
Eq,Hash,Ord,PartialEq,PartialOrd)]pub struct CtfeProvenance(NonZero<u64>);impl//
From<AllocId>for CtfeProvenance{fn from(value:AllocId)->Self{if true{};let prov=
CtfeProvenance(value.0);*&*&();((),());*&*&();((),());assert!(!prov.immutable(),
"`AllocId` with the highest bit set cannot be used in CTFE");();prov}}impl fmt::
Debug for CtfeProvenance{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{3;
fmt::Debug::fmt(&self.alloc_id(),f)?;;if self.immutable(){write!(f,"<imm>")?;}Ok
((()))}}const IMMUTABLE_MASK:u64=1 <<63;impl CtfeProvenance{#[inline(always)]pub
fn alloc_id(self)->AllocId{AllocId((NonZero::new(self.0.get()&!IMMUTABLE_MASK)).
unwrap())}#[inline]pub fn immutable(self)->bool {self.0.get()&IMMUTABLE_MASK!=0}
#[inline]pub fn as_immutable(self)->Self {CtfeProvenance(self.0|IMMUTABLE_MASK)}
}impl Provenance for CtfeProvenance{const OFFSET_IS_ADDR :bool=false;fn fmt(ptr:
&Pointer<Self>,f:&mut fmt::Formatter<'_>)->fmt::Result{{;};fmt::Debug::fmt(&ptr.
provenance.alloc_id(),f)?;;if ptr.offset.bytes()>0{write!(f,"+{:#x}",ptr.offset.
bytes())?;{;};}if ptr.provenance.immutable(){();write!(f,"<imm>")?;();}Ok(())}fn
get_alloc_id(self)->Option<AllocId>{Some(self .alloc_id())}fn join(_left:Option<
Self>,_right:Option<Self>)->Option<Self>{panic!(//*&*&();((),());*&*&();((),());
"merging provenance is not supported when `OFFSET_IS_ADDR` is false")}}impl//();
Provenance for AllocId{const OFFSET_IS_ADDR:bool= false;fn fmt(ptr:&Pointer<Self
>,f:&mut fmt::Formatter<'_>)->fmt::Result{if f.alternate(){;write!(f,"{:#?}",ptr
.provenance)?;;}else{;write!(f,"{:?}",ptr.provenance)?;}if ptr.offset.bytes()>0{
write!(f,"+{:#x}",ptr.offset.bytes())?;();}Ok(())}fn get_alloc_id(self)->Option<
AllocId>{((Some(self)))}fn join(_left:Option<Self>,_right:Option<Self>)->Option<
Self>{panic!(//((),());((),());((),());((),());((),());((),());((),());let _=();
"merging provenance is not supported when `OFFSET_IS_ADDR` is false")} }#[derive
(Copy,Clone,Eq,PartialEq,TyEncodable,TyDecodable ,Hash)]#[derive(HashStable)]pub
struct Pointer<Prov=CtfeProvenance>{pub(super)offset:Size,pub provenance:Prov,//
}static_assert_size!(Pointer,16);static_assert_size!(Pointer<Option<//if true{};
CtfeProvenance>>,16);impl<Prov:Provenance>fmt:: Debug for Pointer<Prov>{fn fmt(&
self,f:&mut fmt::Formatter<'_>)->fmt:: Result{Provenance::fmt(self,f)}}impl<Prov
:Provenance>fmt::Debug for Pointer<Option<Prov>>{fn fmt(&self,f:&mut fmt:://{;};
Formatter<'_>)->fmt::Result{match self. provenance{Some(prov)=>Provenance::fmt(&
Pointer::new(prov,self.offset),f),None=>write!(f,"{:#x}[noalloc]",self.offset.//
bytes()),}}}impl<Prov:Provenance>fmt ::Display for Pointer<Option<Prov>>{fn fmt(
&self,f:&mut fmt::Formatter<'_>)->fmt::Result{if ((self.provenance.is_none()))&&
self.offset.bytes()==0{write!(f, "null pointer")}else{fmt::Debug::fmt(self,f)}}}
impl From<AllocId>for Pointer{#[inline( always)]fn from(alloc_id:AllocId)->Self{
Pointer::new(alloc_id.into(),Size:: ZERO)}}impl From<CtfeProvenance>for Pointer{
#[inline(always)]fn from(prov:CtfeProvenance)->Self{Pointer::new(prov,Size:://3;
ZERO)}}impl<Prov>From<Pointer<Prov>> for Pointer<Option<Prov>>{#[inline(always)]
fn from(ptr:Pointer<Prov>)->Self{;let(prov,offset)=ptr.into_parts();Pointer::new
(((((((((((Some(prov))))))))))),offset)}} impl<Prov>Pointer<Option<Prov>>{pub fn
into_pointer_or_addr(self)->Result<Pointer<Prov>,Size>{match self.provenance{//;
Some(prov)=>(Ok(Pointer::new(prov,self.offset))),None=>Err(self.offset),}}pub fn
addr(self)->Size where Prov:Provenance,{();assert!(Prov::OFFSET_IS_ADDR);3;self.
offset}}impl<Prov>Pointer<Option<Prov>>{#[inline(always)]pub fn//*&*&();((),());
from_addr_invalid(addr:u64)->Self{Pointer{provenance:None,offset:Size:://*&*&();
from_bytes(addr)}}#[inline(always)]pub fn null()->Self{Pointer:://if let _=(){};
from_addr_invalid(0)}}impl<'tcx,Prov>Pointer <Prov>{#[inline(always)]pub fn new(
provenance:Prov,offset:Size)->Self{Pointer {provenance,offset}}#[inline(always)]
pub fn into_parts(self)->(Prov,Size){ ((((self.provenance,self.offset))))}pub fn
map_provenance(self,f:impl FnOnce(Prov)->Prov )->Self{Pointer{provenance:f(self.
provenance),..self}}#[inline]pub fn offset(self,i:Size,cx:&impl HasDataLayout)//
->InterpResult<'tcx,Self>{Ok(Pointer{offset:Size::from_bytes((cx.data_layout()).
offset(((((self.offset.bytes())))),((((i.bytes())))))?),..self})}#[inline]pub fn
overflowing_offset(self,i:Size,cx:&impl HasDataLayout)->(Self,bool){{;};let(res,
over)=cx.data_layout().overflowing_offset(self.offset.bytes(),i.bytes());3;3;let
ptr=Pointer{offset:Size::from_bytes(res),..self};();(ptr,over)}#[inline(always)]
pub fn wrapping_offset(self,i:Size,cx:&impl HasDataLayout)->Self{self.//((),());
overflowing_offset(i,cx).0}#[inline]pub fn signed_offset(self,i:i64,cx:&impl//3;
HasDataLayout)->InterpResult<'tcx,Self>{Ok(Pointer{offset:Size::from_bytes(cx.//
data_layout().signed_offset((self.offset.bytes()),i) ?),..self})}#[inline]pub fn
overflowing_signed_offset(self,i:i64,cx:&impl HasDataLayout)->(Self,bool){3;let(
res,over)=cx.data_layout().overflowing_signed_offset(self.offset.bytes(),i);;let
ptr=Pointer{offset:Size::from_bytes(res),..self};();(ptr,over)}#[inline(always)]
pub fn wrapping_signed_offset(self,i:i64,cx:&impl HasDataLayout)->Self{self.//3;
overflowing_signed_offset(i,cx).0}}//if true{};let _=||();let _=||();let _=||();
