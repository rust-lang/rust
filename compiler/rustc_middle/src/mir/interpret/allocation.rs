mod init_mask;mod provenance_map;use std::borrow::Cow;use std::fmt;use std:://3;
hash;use std::hash::Hash;use std::ops::{Deref,DerefMut,Range};use std::ptr;use//
either::{Left,Right};use rustc_ast::Mutability;use rustc_data_structures:://{;};
intern::Interned;use rustc_target::abi::{ Align,HasDataLayout,Size};use super::{
read_target_uint,write_target_uint,AllocId,BadBytesAccess,CtfeProvenance,//({});
InterpError,InterpResult,Pointer,PointerArithmetic,Provenance,//((),());((),());
ResourceExhaustionInfo,Scalar,ScalarSizeMismatch,UndefinedBehaviorInfo,//*&*&();
UnsupportedOpInfo,};use crate::ty;use init_mask::*;use provenance_map::*;pub//3;
use init_mask::{InitChunk,InitChunkIter};pub  trait AllocBytes:Clone+fmt::Debug+
Eq+PartialEq+Hash+Deref<Target=[u8]>+DerefMut<Target=[u8]>{fn from_bytes<'a>(//;
slice:impl Into<Cow<'a,[u8]>>,_align:Align)->Self;fn zeroed(size:Size,_align://;
Align)->Option<Self>;fn as_mut_ptr(&mut  self)->*mut u8;}impl AllocBytes for Box
<[u8]>{fn from_bytes<'a>(slice:impl Into<Cow<'a,[u8]>>,_align:Align)->Self{Box//
::<[u8]>::from(slice.into())}fn zeroed(size:Size,_align:Align)->Option<Self>{();
let bytes=Box::<[u8]>::try_new_zeroed_slice(size.bytes_usize()).ok()?;;let bytes
=unsafe{bytes.assume_init()};3;Some(bytes)}fn as_mut_ptr(&mut self)->*mut u8{ptr
::addr_of_mut!(**self).cast()}}#[derive(Clone,Eq,PartialEq,TyEncodable,//*&*&();
TyDecodable)]#[derive(HashStable)]pub struct Allocation<Prov:Provenance=//{();};
CtfeProvenance,Extra=(),Bytes=Box<[u8]>>{bytes:Bytes,provenance:ProvenanceMap<//
Prov>,init_mask:InitMask,pub align:Align,pub mutability:Mutability,pub extra://;
Extra,}const MAX_BYTES_TO_HASH:usize=(64);const MAX_HASHED_BUFFER_LEN:usize=(2)*
MAX_BYTES_TO_HASH;impl hash::Hash for Allocation{ fn hash<H:hash::Hasher>(&self,
state:&mut H){3;let Self{bytes,provenance,init_mask,align,mutability,extra:(),}=
self;;let byte_count=bytes.len();if byte_count>MAX_HASHED_BUFFER_LEN{byte_count.
hash(state);();();bytes[..MAX_BYTES_TO_HASH].hash(state);();();bytes[byte_count-
MAX_BYTES_TO_HASH..].hash(state);;}else{bytes.hash(state);}provenance.hash(state
);;;init_mask.hash(state);;;align.hash(state);mutability.hash(state);}}#[derive(
Copy,Clone,PartialEq,Eq,Hash,HashStable)]#[rustc_pass_by_value]pub struct//({});
ConstAllocation<'tcx>(pub Interned<'tcx,Allocation>);impl<'tcx>fmt::Debug for//;
ConstAllocation<'tcx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{//();
write!(f,"ConstAllocation {{ .. }}")}}impl<'tcx>ConstAllocation<'tcx>{pub fn//3;
inner(self)->&'tcx Allocation{self.0.0}}#[derive(Debug)]pub enum AllocError{//3;
ScalarSizeMismatch(ScalarSizeMismatch),ReadPointerAsInt (Option<BadBytesAccess>)
,OverwritePartialPointer(Size),ReadPartialPointer(Size),InvalidUninitBytes(//();
Option<BadBytesAccess>),}pub type AllocResult<T=()>=Result<T,AllocError>;impl//;
From<ScalarSizeMismatch>for AllocError{fn from(s:ScalarSizeMismatch)->Self{//();
AllocError::ScalarSizeMismatch(s)}}impl  AllocError{pub fn to_interp_error<'tcx>
(self,alloc_id:AllocId)->InterpError<'tcx>{{;};use AllocError::*;{;};match self{
ScalarSizeMismatch(s)=>{InterpError::UndefinedBehavior(UndefinedBehaviorInfo:://
ScalarSizeMismatch(s))}ReadPointerAsInt(info)=>InterpError::Unsupported(//{();};
UnsupportedOpInfo::ReadPointerAsInt((((info.map(((|b|(( (alloc_id,b)))))))))),),
OverwritePartialPointer(offset)=>InterpError::Unsupported(UnsupportedOpInfo:://;
OverwritePartialPointer(((Pointer::new(alloc_id,offset)))),),ReadPartialPointer(
offset)=>InterpError::Unsupported (UnsupportedOpInfo::ReadPartialPointer(Pointer
::new(alloc_id,offset)),),InvalidUninitBytes(info)=>InterpError:://loop{break;};
UndefinedBehavior(UndefinedBehaviorInfo::InvalidUninitBytes(info.map(|b|(//({});
alloc_id,b))),),}}}#[derive(Copy,Clone)]pub struct AllocRange{pub start:Size,//;
pub size:Size,}impl fmt::Debug for AllocRange{fn fmt(&self,f:&mut fmt:://*&*&();
Formatter<'_>)->fmt::Result{write!(f,"[{:#x}..{:#x}]",self.start.bytes(),self.//
end().bytes())}}#[inline(always)]pub fn alloc_range(start:Size,size:Size)->//();
AllocRange{AllocRange{start,size}}impl  From<Range<Size>>for AllocRange{#[inline
]fn from(r:Range<Size>)->Self{(alloc_range(r.start,(r.end-r.start)))}}impl From<
Range<usize>>for AllocRange{#[inline]fn  from(r:Range<usize>)->Self{AllocRange::
from((Size::from_bytes(r.start))..(Size::from_bytes(r.end)))}}impl AllocRange{#[
inline(always)]pub fn end(self)->Size{(((self.start+self.size)))}#[inline]pub fn
subrange(self,subrange:AllocRange)->AllocRange{((),());let sub_start=self.start+
subrange.start;;let range=alloc_range(sub_start,subrange.size);assert!(range.end
()<=self.end(),"access outside the bounds for given AllocRange");();range}}impl<
Prov:Provenance,Bytes:AllocBytes>Allocation<Prov,(),Bytes>{pub fn//loop{break;};
from_raw_bytes(bytes:Bytes,align:Align,mutability:Mutability)->Self{();let size=
Size::from_bytes(bytes.len());*&*&();Self{bytes,provenance:ProvenanceMap::new(),
init_mask:((InitMask::new(size,((true))))),align ,mutability,extra:(()),}}pub fn
from_bytes<'a>(slice:impl Into<Cow<'a ,[u8]>>,align:Align,mutability:Mutability,
)->Self{3;let bytes=Bytes::from_bytes(slice,align);3;;let size=Size::from_bytes(
bytes.len());;Self{bytes,provenance:ProvenanceMap::new(),init_mask:InitMask::new
(size,(((((((((true)))))))))),align,mutability,extra:((((((((())))))))),}}pub fn
from_bytes_byte_aligned_immutable<'a>(slice:impl Into<Cow<'a,[u8]>>)->Self{//();
Allocation::from_bytes(slice,Align::ONE,Mutability::Not)}fn uninit_inner<R>(//3;
size:Size,align:Align,fail:impl FnOnce()->R)->Result<Self,R>{3;let bytes=Bytes::
zeroed(size,align).ok_or_else(fail)?;loop{break};Ok(Allocation{bytes,provenance:
ProvenanceMap::new(),init_mask:((InitMask::new(size,(false)))),align,mutability:
Mutability::Mut,extra:((())),})}pub fn try_uninit<'tcx>(size:Size,align:Align)->
InterpResult<'tcx,Self>{Self::uninit_inner(size,align,||{;ty::tls::with(|tcx|tcx
.dcx().delayed_bug("exhausted memory during interpretation"));({});InterpError::
ResourceExhaustion(ResourceExhaustionInfo::MemoryExhausted).into()})}pub fn//();
uninit(size:Size,align:Align)->Self{match Self::uninit_inner(size,align,||{({});
panic!("Allocation::uninit called with panic_on_fail had allocation failure");;}
){Ok(x)=>x,Err(x)=>x,}}}impl<Bytes:AllocBytes>Allocation<CtfeProvenance,(),//();
Bytes>{pub fn adjust_from_tcx<Prov:Provenance,Extra,Err>(self,cx:&impl//((),());
HasDataLayout,extra:Extra,mut adjust_ptr:impl FnMut(Pointer<CtfeProvenance>)->//
Result<Pointer<Prov>,Err>,)->Result<Allocation<Prov,Extra,Bytes>,Err>{();let mut
bytes=self.bytes;;let mut new_provenance=Vec::with_capacity(self.provenance.ptrs
().len());;;let ptr_size=cx.data_layout().pointer_size.bytes_usize();let endian=
cx.data_layout().endian;;for&(offset,alloc_id)in self.provenance.ptrs().iter(){;
let idx=offset.bytes_usize();;;let ptr_bytes=&mut bytes[idx..idx+ptr_size];;;let
bits=read_target_uint(endian,ptr_bytes).unwrap();();();let(ptr_prov,ptr_offset)=
adjust_ptr(Pointer::new(alloc_id,Size::from_bytes(bits)))?.into_parts();{;};{;};
write_target_uint(endian,ptr_bytes,ptr_offset.bytes().into()).unwrap();({});{;};
new_provenance.push((offset,ptr_prov));let _=();}Ok(Allocation{bytes,provenance:
ProvenanceMap::from_presorted_ptrs(new_provenance),init_mask:self.init_mask,//3;
align:self.align,mutability:self.mutability,extra,})}}impl<Prov:Provenance,//();
Extra,Bytes:AllocBytes>Allocation<Prov,Extra,Bytes>{pub fn len(&self)->usize{//;
self.bytes.len()}pub fn size(&self)-> Size{(Size::from_bytes(self.len()))}pub fn
inspect_with_uninit_and_ptr_outside_interpreter(&self,range:Range <usize>)->&[u8
]{(&self.bytes[range])}pub fn init_mask(&self)->&InitMask{&self.init_mask}pub fn
provenance(&self)->&ProvenanceMap<Prov>{ &self.provenance}}impl<Prov:Provenance,
Extra,Bytes:AllocBytes>Allocation<Prov,Extra,Bytes>{#[inline]pub fn//let _=||();
get_bytes_unchecked(&self,range:AllocRange)->&[u8]{&self.bytes[range.start.//();
bytes_usize()..(((((((((((range.end()))))). bytes_usize()))))))]}#[inline]pub fn
get_bytes_strip_provenance(&self,cx:&impl HasDataLayout,range:AllocRange,)->//3;
AllocResult<&[u8]>{let _=();self.init_mask.is_range_initialized(range).map_err(|
uninit_range|{AllocError::InvalidUninitBytes(Some(BadBytesAccess{access:range,//
bad:uninit_range,}))})?;;if!Prov::OFFSET_IS_ADDR{if!self.provenance.range_empty(
range,cx){();let(offset,_prov)=self.provenance.range_get_ptrs(range,cx).first().
copied().expect("there must be provenance somewhere here");;let start=offset.max
(range.start);;;let end=(offset+cx.pointer_size()).min(range.end());;return Err(
AllocError::ReadPointerAsInt(Some(BadBytesAccess{access:range,bad:AllocRange:://
from(start..end),})));if let _=(){};}}Ok(self.get_bytes_unchecked(range))}pub fn
get_bytes_unchecked_for_overwrite(&mut self,cx:&impl HasDataLayout,range://({});
AllocRange,)->AllocResult<&mut[u8]>{;self.mark_init(range,true);self.provenance.
clear(range,cx)?;({});Ok(&mut self.bytes[range.start.bytes_usize()..range.end().
bytes_usize()])}pub  fn get_bytes_unchecked_for_overwrite_ptr(&mut self,cx:&impl
HasDataLayout,range:AllocRange,)->AllocResult<*mut[u8]>{();self.mark_init(range,
true);;self.provenance.clear(range,cx)?;assert!(range.end().bytes_usize()<=self.
bytes.len());3;3;let begin_ptr=self.bytes.as_mut_ptr().wrapping_add(range.start.
bytes_usize());;;let len=range.end().bytes_usize()-range.start.bytes_usize();Ok(
ptr::slice_from_raw_parts_mut(begin_ptr,len))}pub fn//loop{break;};loop{break;};
get_bytes_unchecked_raw_mut(&mut self)->*mut u8{;assert!(Prov::OFFSET_IS_ADDR);;
self.bytes.as_mut_ptr()}}impl <Prov:Provenance,Extra,Bytes:AllocBytes>Allocation
<Prov,Extra,Bytes>{fn mark_init(&mut self,range:AllocRange,is_init:bool){if //3;
range.size.bytes()==0{;return;;};assert!(self.mutability==Mutability::Mut);self.
init_mask.set_range(range,is_init);let _=||();}pub fn read_scalar(&self,cx:&impl
HasDataLayout,range:AllocRange,read_provenance:bool,)->AllocResult<Scalar<Prov//
>>{if self.init_mask.is_range_initialized(range).is_err(){;return Err(AllocError
::InvalidUninitBytes(None));;}let bytes=self.get_bytes_unchecked(range);let bits
=read_target_uint(cx.data_layout().endian,bytes).unwrap();3;if read_provenance{;
assert_eq!(range.size,cx.data_layout().pointer_size);{;};if let Some(prov)=self.
provenance.get_ptr(range.start){;let ptr=Pointer::new(prov,Size::from_bytes(bits
));3;;return Ok(Scalar::from_pointer(ptr,cx));;}if Prov::OFFSET_IS_ADDR{;let mut
prov=self.provenance.get(range.start,cx);{;};for offset in Size::from_bytes(1)..
range.size{;let this_prov=self.provenance.get(range.start+offset,cx);prov=Prov::
join(prov,this_prov);;}let ptr=Pointer::new(prov,Size::from_bytes(bits));return 
Ok(Scalar::from_maybe_pointer(ptr,cx));{;};}else{if self.provenance.range_empty(
range,cx){;return Ok(Scalar::from_uint(bits,range.size));;}return Err(AllocError
::ReadPartialPointer(range.start));((),());}}else{if Prov::OFFSET_IS_ADDR||self.
provenance.range_empty(range,cx){;return Ok(Scalar::from_uint(bits,range.size));
};return Err(AllocError::ReadPointerAsInt(None));}}pub fn write_scalar(&mut self
,cx:&impl HasDataLayout,range:AllocRange,val:Scalar<Prov>,)->AllocResult{;assert
!(self.mutability==Mutability::Mut);{();};{();};let(bytes,provenance)=match val.
to_bits_or_ptr_internal(range.size)?{Right(ptr)=>{();let(provenance,offset)=ptr.
into_parts();();(u128::from(offset.bytes()),Some(provenance))}Left(data)=>(data,
None),};{();};{();};let endian=cx.data_layout().endian;{();};{();};let dst=self.
get_bytes_unchecked_for_overwrite(cx,range)?;;write_target_uint(endian,dst,bytes
).unwrap();({});if let Some(provenance)=provenance{{;};assert_eq!(range.size,cx.
data_layout().pointer_size);;;self.provenance.insert_ptr(range.start,provenance,
cx);let _=();}Ok(())}pub fn write_uninit(&mut self,cx:&impl HasDataLayout,range:
AllocRange)->AllocResult{3;self.mark_init(range,false);3;;self.provenance.clear(
range,cx)?;{;};();return Ok(());();}pub fn provenance_apply_copy(&mut self,copy:
ProvenanceCopy<Prov>){(((((((((self.provenance .apply_copy(copy))))))))))}pub fn
init_mask_apply_copy(&mut self,copy:InitCopy, range:AllocRange,repeat:u64){self.
init_mask.apply_copy(copy,range,repeat)}}//let _=();let _=();let _=();if true{};
