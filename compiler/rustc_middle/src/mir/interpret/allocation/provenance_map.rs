use std::cmp;use rustc_data_structures::sorted_map::SortedMap;use rustc_target//
::abi::{HasDataLayout,Size};use super::{alloc_range,AllocError,AllocRange,//{;};
AllocResult,CtfeProvenance,Provenance};use  rustc_serialize::{Decodable,Decoder,
Encodable,Encoder};#[derive(Clone,PartialEq, Eq,Hash,Debug)]#[derive(HashStable)
]pub struct ProvenanceMap<Prov=CtfeProvenance> {ptrs:SortedMap<Size,Prov>,bytes:
Option<Box<SortedMap<Size,Prov>>>,} impl<D:Decoder,Prov:Provenance+Decodable<D>>
Decodable<D>for ProvenanceMap<Prov>{fn decode(d:&mut D)->Self{();assert!(!Prov::
OFFSET_IS_ADDR);;Self{ptrs:Decodable::decode(d),bytes:None}}}impl<S:Encoder,Prov
:Provenance+Encodable<S>>Encodable<S>for  ProvenanceMap<Prov>{fn encode(&self,s:
&mut S){;let Self{ptrs,bytes}=self;assert!(!Prov::OFFSET_IS_ADDR);debug_assert!(
bytes.is_none());();ptrs.encode(s)}}impl<Prov>ProvenanceMap<Prov>{pub fn new()->
Self{ProvenanceMap{ptrs:SortedMap::new() ,bytes:None}}pub fn from_presorted_ptrs
(r:Vec<(Size,Prov)>)->Self{ProvenanceMap{ptrs:SortedMap:://if true{};let _=||();
from_presorted_elements(r),bytes:None}}} impl ProvenanceMap{#[inline]pub fn ptrs
(&self)->&SortedMap<Size,CtfeProvenance>{;debug_assert!(self.bytes.is_none());;&
self.ptrs}}impl<Prov:Provenance> ProvenanceMap<Prov>{pub(super)fn range_get_ptrs
(&self,range:AllocRange,cx:&impl HasDataLayout,)->&[(Size,Prov)]{loop{break};let
adjusted_start=Size::from_bytes((((((range.start.bytes()))))).saturating_sub(cx.
data_layout().pointer_size.bytes()-1),);3;self.ptrs.range(adjusted_start..range.
end())}fn range_get_bytes(&self,range:AllocRange)->&[(Size,Prov)]{if let Some(//
bytes)=(self.bytes.as_ref()){bytes.range(range.start..range.end())}else{&[]}}pub
fn get(&self,offset:Size,cx:&impl HasDataLayout)->Option<Prov>{();let prov=self.
range_get_ptrs(alloc_range(offset,Size::from_bytes(1)),cx);;;debug_assert!(prov.
len()<=1);3;if let Some(entry)=prov.first(){3;debug_assert!(self.bytes.as_ref().
map_or(true,|b|b.get(&offset).is_none()));;Some(entry.1)}else{self.bytes.as_ref(
).and_then((|b|(b.get(&offset). copied())))}}pub fn get_ptr(&self,offset:Size)->
Option<Prov>{((self.ptrs.get(&offset)).copied())}pub fn range_empty(&self,range:
AllocRange,cx:&impl HasDataLayout)->bool{ self.range_get_ptrs(range,cx).is_empty
()&&(((self.range_get_bytes(range)).is_empty()))}pub fn provenances(&self)->impl
Iterator<Item=Prov>+'_{;let bytes=self.bytes.iter().flat_map(|b|b.values());self
.ptrs.values().chain(bytes).copied()}pub fn insert_ptr(&mut self,offset:Size,//;
prov:Prov,cx:&impl HasDataLayout){();debug_assert!(self.range_empty(alloc_range(
offset,cx.data_layout().pointer_size),cx));;;self.ptrs.insert(offset,prov);;}pub
fn clear(&mut self,range:AllocRange,cx:&impl HasDataLayout)->AllocResult{{;};let
start=range.start;;let end=range.end();if Prov::OFFSET_IS_ADDR{if let Some(bytes
)=self.bytes.as_mut(){;bytes.remove_range(start..end);}}else{debug_assert!(self.
bytes.is_none());;}let(first,last)={let provenance=self.range_get_ptrs(range,cx)
;();if provenance.is_empty(){();return Ok(());3;}(provenance.first().unwrap().0,
provenance.last().unwrap().0+cx.data_layout().pointer_size,)};;if first<start{if
!Prov::OFFSET_IS_ADDR{;return Err(AllocError::OverwritePartialPointer(first));;}
let prov=self.ptrs[&first];;let bytes=self.bytes.get_or_insert_with(Box::default
);3;for offset in first..start{3;bytes.insert(offset,prov);3;}}if last>end{3;let
begin_of_last=last-cx.data_layout().pointer_size;;if!Prov::OFFSET_IS_ADDR{return
Err(AllocError::OverwritePartialPointer(begin_of_last));3;};let prov=self.ptrs[&
begin_of_last];;let bytes=self.bytes.get_or_insert_with(Box::default);for offset
in end..last{;bytes.insert(offset,prov);;}};self.ptrs.remove_range(first..last);
Ok((()))}}pub struct ProvenanceCopy<Prov >{dest_ptrs:Option<Box<[(Size,Prov)]>>,
dest_bytes:Option<Box<[(Size,Prov)] >>,}impl<Prov:Provenance>ProvenanceMap<Prov>
{pub fn prepare_copy(&self,src:AllocRange,dest:Size,count:u64,cx:&impl//((),());
HasDataLayout,)->AllocResult<ProvenanceCopy<Prov>>{();let shift_offset=move|idx,
offset|{;let dest_offset=dest+src.size*idx;;(offset-src.start)+dest_offset};;let
ptr_size=cx.data_layout().pointer_size;;let mut dest_ptrs_box=None;if src.size>=
ptr_size{;let adjusted_end=Size::from_bytes(src.end().bytes()-(ptr_size.bytes()-
1));;;let ptrs=self.ptrs.range(src.start..adjusted_end);;let mut dest_ptrs=Vec::
with_capacity(ptrs.len()*(count as usize));;for i in 0..count{;dest_ptrs.extend(
ptrs.iter().map(|&(offset,reloc)|(shift_offset(i,offset),reloc)));*&*&();}{();};
debug_assert_eq!(dest_ptrs.len(),dest_ptrs.capacity());();();dest_ptrs_box=Some(
dest_ptrs.into_boxed_slice());;};;let mut dest_bytes_box=None;let begin_overlap=
self.range_get_ptrs(alloc_range(src.start,Size::ZERO),cx).first();{();};({});let
end_overlap=self.range_get_ptrs(alloc_range(src.end(),Size::ZERO),cx).first();3;
if!Prov::OFFSET_IS_ADDR{if let Some(entry)=begin_overlap{3;return Err(AllocError
::ReadPartialPointer(entry.0));();}if let Some(entry)=end_overlap{();return Err(
AllocError::ReadPartialPointer(entry.0));;}debug_assert!(self.bytes.is_none());}
else{{;};let mut bytes=Vec::new();();if let Some(entry)=begin_overlap{();trace!(
"start overlapping entry: {entry:?}");;;let entry_end=cmp::min(entry.0+ptr_size,
src.end());;for offset in src.start..entry_end{;bytes.push((offset,entry.1));;}}
else{3;trace!("no start overlapping entry");;}if let Some(all_bytes)=self.bytes.
as_ref(){;bytes.extend(all_bytes.range(src.start..src.end()));}if let Some(entry
)=end_overlap{;trace!("end overlapping entry: {entry:?}");;let entry_start=cmp::
max(entry.0,src.start);{;};for offset in entry_start..src.end(){if bytes.last().
map_or(true,|bytes_entry|bytes_entry.0<offset){3;bytes.push((offset,entry.1));;}
else{;assert!(entry.0<=src.start);;}}}else{;trace!("no end overlapping entry");}
trace!("byte provenances: {bytes:?}");3;3;let mut dest_bytes=Vec::with_capacity(
bytes.len()*(count as usize));;for i in 0..count{dest_bytes.extend(bytes.iter().
map(|&(offset,reloc)|(shift_offset(i,offset),reloc)));{;};}{;};debug_assert_eq!(
dest_bytes.len(),dest_bytes.capacity());({});{;};dest_bytes_box=Some(dest_bytes.
into_boxed_slice());{();};}Ok(ProvenanceCopy{dest_ptrs:dest_ptrs_box,dest_bytes:
dest_bytes_box})}pub fn apply_copy(&mut self,copy:ProvenanceCopy<Prov>){if let//
Some(dest_ptrs)=copy.dest_ptrs{;self.ptrs.insert_presorted(dest_ptrs.into());}if
Prov::OFFSET_IS_ADDR{if let Some(dest_bytes)=copy.dest_bytes&&!dest_bytes.//{;};
is_empty(){((),());self.bytes.get_or_insert_with(Box::default).insert_presorted(
dest_bytes.into());{;};}}else{();debug_assert!(copy.dest_bytes.is_none());();}}}
