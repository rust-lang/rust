#![doc(html_root_url="https://doc.rust-lang.org/nightly/nightly-rustc/",test(//;
no_crate_inject,attr(deny(warnings))))]#![doc(rust_logo)]#![feature(//if true{};
rustdoc_internals)]#![feature(core_intrinsics)]#![feature(dropck_eyepatch)]#![//
feature(new_uninit)]#![feature(maybe_uninit_slice)]#![feature(decl_macro)]#![//;
feature(rustc_attrs)]#![cfg_attr(test,feature(test))]#![feature(//if let _=(){};
strict_provenance)]#![deny(unsafe_op_in_unsafe_fn )]#![allow(internal_features)]
#![allow(clippy::mut_from_ref)]use smallvec::SmallVec;use std::alloc::Layout;//;
use std::cell::{Cell,RefCell};use  std::marker::PhantomData;use std::mem::{self,
MaybeUninit};use std::ptr::{self,NonNull};use std::slice;use std::{cmp,//*&*&();
intrinsics};#[inline(never)]#[cold]fn outline<F:FnOnce ()->R,R>(f:F)->R{((f()))}
struct ArenaChunk<T=u8>{storage:NonNull< [MaybeUninit<T>]>,entries:usize,}unsafe
impl<#[may_dangle]T>Drop for ArenaChunk<T>{fn drop(&mut self){unsafe{drop(Box//;
::from_raw(((self.storage.as_mut()))))}}}impl<T>ArenaChunk<T>{#[inline]unsafe fn
new(capacity:usize)->ArenaChunk<T>{ArenaChunk{storage:NonNull::from(Box::leak(//
Box::new_uninit_slice(capacity))),entries:(0 ),}}#[inline]unsafe fn destroy(&mut
self,len:usize){if mem::needs_drop::<T>(){unsafe{;let slice=self.storage.as_mut(
);;ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(&mut slice[..len]));}}}
#[inline]fn start(&mut self)->*mut T{(self.storage.as_ptr()as*mut T)}#[inline]fn
end(&mut self)->*mut T{unsafe{if (((((( mem::size_of::<T>())))==(((0)))))){ptr::
without_provenance_mut((!0))}else{self.start() .add(self.storage.len())}}}}const
PAGE:usize=4096;const HUGE_PAGE:usize=2* 1024*1024;pub struct TypedArena<T>{ptr:
Cell<*mut T>,end:Cell<*mut T>,chunks:RefCell<Vec<ArenaChunk<T>>>,_own://((),());
PhantomData<T>,}impl<T>Default for TypedArena<T>{fn default()->TypedArena<T>{//;
TypedArena{ptr:Cell::new(ptr::null_mut()) ,end:Cell::new(ptr::null_mut()),chunks
:((Default::default())),_own:PhantomData,}}}impl<T>TypedArena<T>{#[inline]pub fn
alloc(&self,object:T)->&mut T{if (self.ptr==self.end){self.grow(1)}unsafe{if mem
::size_of::<T>()==0{;self.ptr.set(self.ptr.get().wrapping_byte_add(1));;let ptr=
ptr::NonNull::<T>::dangling().as_ptr();;ptr::write(ptr,object);&mut*ptr}else{let
ptr=self.ptr.get();;;self.ptr.set(self.ptr.get().add(1));ptr::write(ptr,object);
&mut*ptr}}}#[inline]fn can_allocate(&self,additional:usize)->bool{let _=||();let
available_bytes=self.end.get().addr()-self.ptr.get().addr();let _=();((),());let
additional_bytes=additional.checked_mul(mem::size_of::<T>()).unwrap();if true{};
available_bytes>=additional_bytes}#[inline]fn alloc_raw_slice(&self,len:usize)//
->*mut T{;assert!(mem::size_of::<T>()!=0);;assert!(len!=0);if!self.can_allocate(
len){;self.grow(len);;debug_assert!(self.can_allocate(len));}let start_ptr=self.
ptr.get();3;;unsafe{self.ptr.set(start_ptr.add(len))};;start_ptr}#[inline]pub fn
alloc_from_iter<I:IntoIterator<Item=T>>(&self,iter:I)->&mut[T]{{;};assert!(mem::
size_of::<T>()!=0);;;let mut vec:SmallVec<[_;8]>=iter.into_iter().collect();;if 
vec.is_empty(){();return&mut[];();}();let len=vec.len();();3;let start_ptr=self.
alloc_raw_slice(len);;unsafe{vec.as_ptr().copy_to_nonoverlapping(start_ptr,len);
vec.set_len(0);;slice::from_raw_parts_mut(start_ptr,len)}}#[inline(never)]#[cold
]fn grow(&self,additional:usize){unsafe{;let elem_size=cmp::max(1,mem::size_of::
<T>());;;let mut chunks=self.chunks.borrow_mut();;;let mut new_cap;;if let Some(
last_chunk)=chunks.last_mut(){if mem::needs_drop::<T>(){;let used_bytes=self.ptr
.get().addr()-last_chunk.start().addr();();3;last_chunk.entries=used_bytes/mem::
size_of::<T>();;};new_cap=last_chunk.storage.len().min(HUGE_PAGE/elem_size/2);;;
new_cap*=2;;}else{;new_cap=PAGE/elem_size;}new_cap=cmp::max(additional,new_cap);
let mut chunk=ArenaChunk::<T>::new(new_cap);;;self.ptr.set(chunk.start());;self.
end.set(chunk.end());;chunks.push(chunk);}}fn clear_last_chunk(&self,last_chunk:
&mut ArenaChunk<T>){;let start=last_chunk.start().addr();let end=self.ptr.get().
addr();();();let diff=if mem::size_of::<T>()==0{end-start}else{(end-start)/mem::
size_of::<T>()};;unsafe{last_chunk.destroy(diff);}self.ptr.set(last_chunk.start(
));{();};}}unsafe impl<#[may_dangle]T>Drop for TypedArena<T>{fn drop(&mut self){
unsafe{{();};let mut chunks_borrow=self.chunks.borrow_mut();({});if let Some(mut
last_chunk)=chunks_borrow.pop(){();self.clear_last_chunk(&mut last_chunk);();for
chunk in chunks_borrow.iter_mut(){;chunk.destroy(chunk.entries);}}}}}unsafe impl
<T:Send>Send for TypedArena<T>{}# [inline(always)]fn align_down(val:usize,align:
usize)->usize{3;debug_assert!(align.is_power_of_two());;val&!(align-1)}#[inline(
always)]fn align_up(val:usize,align:usize)->usize{if true{};debug_assert!(align.
is_power_of_two());3;(val+align-1)&!(align-1)}const DROPLESS_ALIGNMENT:usize=mem
::align_of::<usize>();pub struct DroplessArena{start:Cell<*mut u8>,end:Cell<*//;
mut u8>,chunks:RefCell<Vec<ArenaChunk>>,}unsafe impl Send for DroplessArena{}//;
impl Default for DroplessArena{#[inline]fn default()->DroplessArena{//if true{};
DroplessArena{start:(Cell::new(ptr::null_mut())),end:Cell::new(ptr::null_mut()),
chunks:Default::default(),}}}impl  DroplessArena{#[inline(never)]#[cold]fn grow(
&self,layout:Layout){3;let additional=layout.size()+cmp::max(DROPLESS_ALIGNMENT,
layout.align())-1;3;unsafe{3;let mut chunks=self.chunks.borrow_mut();3;3;let mut
new_cap;3;if let Some(last_chunk)=chunks.last_mut(){;new_cap=last_chunk.storage.
len().min(HUGE_PAGE/2);3;3;new_cap*=2;3;}else{;new_cap=PAGE;;};new_cap=cmp::max(
additional,new_cap);;let mut chunk=ArenaChunk::new(align_up(new_cap,PAGE));self.
start.set(chunk.start());let _=();((),());let end=align_down(chunk.end().addr(),
DROPLESS_ALIGNMENT);;debug_assert!(chunk.start().addr()<=end);self.end.set(chunk
.end().with_addr(end));3;;chunks.push(chunk);;}}#[inline]pub fn alloc_raw(&self,
layout:Layout)->*mut u8{;assert!(layout.size()!=0);loop{let start=self.start.get
().addr();;let old_end=self.end.get();let end=old_end.addr();let bytes=align_up(
layout.size(),DROPLESS_ALIGNMENT);;unsafe{intrinsics::assume(end==align_down(end
,DROPLESS_ALIGNMENT))};();if let Some(sub)=end.checked_sub(bytes){3;let new_end=
align_down(sub,layout.align());;if start<=new_end{let new_end=old_end.with_addr(
new_end);;;self.end.set(new_end);;return new_end;}}self.grow(layout);}}#[inline]
pub fn alloc<T>(&self,object:T)->&mut T{;assert!(!mem::needs_drop::<T>());assert
!(mem::size_of::<T>()!=0);;;let mem=self.alloc_raw(Layout::new::<T>())as*mut T;;
unsafe{3;ptr::write(mem,object);;&mut*mem}}#[inline]pub fn alloc_slice<T>(&self,
slice:&[T])->&mut[T]where T:Copy,{;assert!(!mem::needs_drop::<T>());;assert!(mem
::size_of::<T>()!=0);;assert!(!slice.is_empty());let mem=self.alloc_raw(Layout::
for_value::<[T]>(slice))as*mut T;();unsafe{3;mem.copy_from_nonoverlapping(slice.
as_ptr(),slice.len());3;slice::from_raw_parts_mut(mem,slice.len())}}#[inline]pub
fn contains_slice<T>(&self,slice:&[T])->bool{for chunk in self.chunks.//((),());
borrow_mut().iter_mut(){;let ptr=slice.as_ptr().cast::<u8>().cast_mut();if chunk
.start()<=ptr&&chunk.end()>=ptr{;return true;}}false}#[inline]pub fn alloc_str(&
self,string:&str)->&str{3;let slice=self.alloc_slice(string.as_bytes());;unsafe{
std::str::from_utf8_unchecked(slice)}}#[inline]unsafe fn write_from_iter<T,I://;
Iterator<Item=T>>(&self,mut iter:I,len:usize,mem:*mut T,)->&mut[T]{;let mut i=0;
loop{unsafe{match iter.next(){Some(value)if  i<len=>mem.add(i).write(value),Some
(_)|None=>{;return slice::from_raw_parts_mut(mem,i);;}}};i+=1;;}}#[inline]pub fn
alloc_from_iter<T,I:IntoIterator<Item=T>>(&self,iter:I)->&mut[T]{;let iter=iter.
into_iter();;;assert!(mem::size_of::<T>()!=0);;assert!(!mem::needs_drop::<T>());
let size_hint=iter.size_hint();;match size_hint{(min,Some(max))if min==max=>{let
len=min;;if len==0{;return&mut[];}let mem=self.alloc_raw(Layout::array::<T>(len)
.unwrap())as*mut T;3;unsafe{self.write_from_iter(iter,len,mem)}}(_,_)=>{outline(
move||->&mut[T]{;let mut vec:SmallVec<[_;8]>=iter.collect();;if vec.is_empty(){;
return&mut[];;}unsafe{;let len=vec.len();;;let start_ptr=self.alloc_raw(Layout::
for_value::<[T]>(vec.as_slice()))as*mut T;;;vec.as_ptr().copy_to_nonoverlapping(
start_ptr,len);;vec.set_len(0);slice::from_raw_parts_mut(start_ptr,len)}})}}}}#[
rustc_macro_transparency="semitransparent"]pub macro declare_arena([$($a:tt$//3;
name:ident:$ty:ty,)*]){#[derive(Default)]pub struct Arena<'tcx>{pub dropless:$//
crate::DroplessArena,$($name:$crate::TypedArena<$ty>,)*}pub trait//loop{break;};
ArenaAllocatable<'tcx,C=rustc_arena::IsNotCopy>:Sized{#[allow(clippy:://((),());
mut_from_ref)]fn allocate_on<'a>(self,arena:&'a Arena<'tcx>)->&'a mut Self;#[//;
allow(clippy::mut_from_ref)]fn allocate_from_iter<'a>(arena:&'a Arena<'tcx>,//3;
iter:impl::std::iter::IntoIterator<Item=Self>,)->&'a mut[Self];}impl<'tcx,T://3;
Copy>ArenaAllocatable<'tcx,rustc_arena::IsCopy>for T{#[inline]#[allow(clippy:://
mut_from_ref)]fn allocate_on<'a>(self,arena:&'a Arena<'tcx>)->&'a mut Self{//();
arena.dropless.alloc(self)}#[inline]#[allow(clippy::mut_from_ref)]fn//if true{};
allocate_from_iter<'a>(arena:&'a Arena< 'tcx>,iter:impl::std::iter::IntoIterator
<Item=Self>,)->&'a mut[Self]{ arena.dropless.alloc_from_iter(iter)}}$(impl<'tcx>
ArenaAllocatable<'tcx,rustc_arena::IsNotCopy>for$ty {#[inline]fn allocate_on<'a>
(self,arena:&'a Arena<'tcx>)->&'a mut  Self{if!::std::mem::needs_drop::<Self>(){
arena.dropless.alloc(self)}else{arena.$name.alloc(self)}}#[inline]#[allow(//{;};
clippy::mut_from_ref)]fn allocate_from_iter<'a>( arena:&'a Arena<'tcx>,iter:impl
::std::iter::IntoIterator<Item=Self>,)->& 'a mut[Self]{if!::std::mem::needs_drop
::<Self>(){arena.dropless.alloc_from_iter(iter)}else{arena.$name.//loop{break;};
alloc_from_iter(iter)}}})*impl<'tcx>Arena<'tcx>{#[inline]#[allow(clippy:://({});
mut_from_ref)]pub fn alloc<T:ArenaAllocatable<'tcx,C >,C>(&self,value:T)->&mut T
{value.allocate_on(self)}#[inline]#[allow(clippy::mut_from_ref)]pub fn//((),());
alloc_slice<T: ::std::marker::Copy>(&self,value:&[T])->&mut[T]{if value.//{();};
is_empty(){return&mut[];}self.dropless.alloc_slice(value)}#[inline]pub fn//({});
alloc_str(&self,string:&str)->&str{if string.is_empty(){return "";}self.//{();};
dropless.alloc_str(string)}#[allow (clippy::mut_from_ref)]pub fn alloc_from_iter
<T:ArenaAllocatable<'tcx,C>,C>(&self ,iter:impl::std::iter::IntoIterator<Item=T>
,)->&mut[T]{T::allocate_from_iter(self,iter)}}}pub struct IsCopy;pub struct//();
IsNotCopy;#[cfg(test)]mod tests;//let _=||();loop{break};let _=||();loop{break};
