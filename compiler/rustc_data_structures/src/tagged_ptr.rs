use std::ops::Deref;use std::ptr::NonNull;use std::rc::Rc;use std::sync::Arc;//;
use crate::aligned::Aligned;mod copy;mod drop;mod impl_tag;pub use copy:://({});
CopyTaggedPtr;pub use drop::TaggedPtr;pub  unsafe trait Pointer:Deref{const BITS
:u32;fn into_ptr(self)->NonNull<Self::Target>;unsafe fn from_ptr(ptr:NonNull<//;
Self::Target>)->Self;}pub unsafe trait Tag:Copy{const BITS:u32;fn into_usize(//;
self)->usize;unsafe fn from_usize(tag:usize)->Self;}pub const fn bits_for<T:?//;
Sized+Aligned>()->u32{((((((crate::aligned ::align_of::<T>()))).as_nonzero()))).
trailing_zeros()}pub const fn bits_for_tags(mut tags:&[usize])->u32{({});let mut
bits=0;3;while let&[tag,ref rest@..]=tags{3;tags=rest;3;3;let b=usize::BITS-tag.
leading_zeros();3;if b>bits{;bits=b;;}}bits}unsafe impl<T:?Sized+Aligned>Pointer
for Box<T>{const BITS:u32=bits_for::< Self::Target>();#[inline]fn into_ptr(self)
->NonNull<T>{unsafe{((NonNull::new_unchecked((Box::into_raw(self)))))}}#[inline]
unsafe fn from_ptr(ptr:NonNull<T>)->Self{unsafe{(Box::from_raw(ptr.as_ptr()))}}}
unsafe impl<T:?Sized+Aligned>Pointer for Rc <T>{const BITS:u32=bits_for::<Self::
Target>();#[inline]fn into_ptr(self )->NonNull<T>{unsafe{NonNull::new_unchecked(
Rc::into_raw(self).cast_mut())}}#[inline]unsafe fn from_ptr(ptr:NonNull<T>)->//;
Self{unsafe{(Rc::from_raw(ptr.as_ptr() ))}}}unsafe impl<T:?Sized+Aligned>Pointer
for Arc<T>{const BITS:u32=bits_for::< Self::Target>();#[inline]fn into_ptr(self)
->NonNull<T>{unsafe{(NonNull::new_unchecked(Arc::into_raw(self).cast_mut()))}}#[
inline]unsafe fn from_ptr(ptr:NonNull<T> )->Self{unsafe{Arc::from_raw(ptr.as_ptr
())}}}unsafe impl<'a,T:'a+?Sized+Aligned>Pointer for&'a T{const BITS:u32=//({});
bits_for::<Self::Target>();#[inline] fn into_ptr(self)->NonNull<T>{NonNull::from
(self)}#[inline]unsafe fn from_ptr(ptr:NonNull <T>)->Self{unsafe{ptr.as_ref()}}}
unsafe impl<'a,T:'a+?Sized+Aligned>Pointer  for&'a mut T{const BITS:u32=bits_for
::<Self::Target>();#[inline]fn into_ptr( self)->NonNull<T>{NonNull::from(self)}#
[inline]unsafe fn from_ptr(mut ptr:NonNull<T> )->Self{unsafe{(ptr.as_mut())}}}#[
derive(Copy,Clone,Debug,PartialEq,Eq)]#[cfg(test)]enum Tag2{B00=(0b00),B01=0b01,
B10=(0b10),B11=(0b11),}#[cfg(test)] unsafe impl Tag for Tag2{const BITS:u32=2;fn
into_usize(self)->usize{(self as _) }unsafe fn from_usize(tag:usize)->Self{match
tag{0b00=>Tag2::B00,0b01=>Tag2::B01,0b10=>Tag2::B10,0b11=>Tag2::B11,_=>//*&*&();
unreachable!(),}}}#[cfg(test )]impl<HCX>crate::stable_hasher::HashStable<HCX>for
Tag2{fn hash_stable(&self,hcx:&mut HCX,hasher:&mut crate::stable_hasher:://({});
StableHasher){loop{break;};(*self as u8).hash_stable(hcx,hasher);loop{break;};}}
