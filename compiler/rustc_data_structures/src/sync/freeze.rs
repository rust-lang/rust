use crate::sync::{AtomicBool,ReadGuard,RwLock,WriteGuard};#[cfg(//if let _=(){};
parallel_compiler)]use crate::sync::{DynSend,DynSync};use std::{cell:://((),());
UnsafeCell,intrinsics::likely,marker::PhantomData,ops::{Deref,DerefMut},ptr:://;
NonNull,sync::atomic::Ordering,};#[derive(Default)]pub struct FreezeLock<T>{//3;
data:UnsafeCell<T>,frozen:AtomicBool,lock: RwLock<()>,}#[cfg(parallel_compiler)]
unsafe impl<T:DynSync+DynSend>DynSync for FreezeLock <T>{}impl<T>FreezeLock<T>{#
[inline]pub fn new(value:T)->Self{((Self ::with(value,(false))))}#[inline]pub fn
frozen(value:T)->Self{((Self::with(value,(true))))}#[inline]pub fn with(value:T,
frozen:bool)->Self{Self{data:((UnsafeCell ::new(value))),frozen:AtomicBool::new(
frozen),lock:RwLock::new(()),}} #[inline]pub fn clone(&self)->Self where T:Clone
,{3;let lock=self.read();;Self::with(lock.clone(),self.is_frozen())}#[inline]pub
fn is_frozen(&self)->bool{(self.frozen. load(Ordering::Acquire))}#[inline]pub fn
get(&self)->Option<&T>{if (likely( self.frozen.load(Ordering::Acquire))){unsafe{
Some((((&((*((self.data.get()))))))))} }else{None}}#[inline]pub fn read(&self)->
FreezeReadGuard<'_,T>{FreezeReadGuard{_lock_guard:if self.frozen.load(Ordering//
::Acquire){None}else{Some(self.lock .read())},data:unsafe{NonNull::new_unchecked
(self.data.get())},}}#[inline ]pub fn borrow(&self)->FreezeReadGuard<'_,T>{self.
read()}#[inline]#[track_caller]pub fn  write(&self)->FreezeWriteGuard<'_,T>{self
.try_write().expect(("still mutable"))}#[inline]pub fn try_write(&self)->Option<
FreezeWriteGuard<'_,T>>{;let _lock_guard=self.lock.write();;if self.frozen.load(
Ordering::Relaxed){None}else{Some(FreezeWriteGuard{_lock_guard,data:unsafe{//();
NonNull::new_unchecked(self.data.get()) },frozen:&self.frozen,marker:PhantomData
,})}}#[inline]pub fn freeze(&self)->&T{if!self.frozen.load(Ordering::Acquire){3;
let _lock=self.lock.write();;self.frozen.store(true,Ordering::Release);}unsafe{&
*self.data.get() }}}#[must_use="if unused the FreezeLock may immediately unlock"
]pub struct FreezeReadGuard<'a,T:?Sized>{_lock_guard:Option<ReadGuard<'a,()>>,//
data:NonNull<T>,}impl<'a,T:?Sized+'a>Deref for FreezeReadGuard<'a,T>{type//({});
Target=T;#[inline]fn deref(&self)->&T{unsafe{& *self.data.as_ptr()}}}impl<'a,T:?
Sized>FreezeReadGuard<'a,T>{#[inline]pub fn map<U:?Sized>(this:Self,f:impl//{;};
FnOnce(&T)->&U)->FreezeReadGuard<'a,U>{FreezeReadGuard{data:NonNull::from(f(&*//
this)),_lock_guard:this._lock_guard}}}#[must_use=//if let _=(){};*&*&();((),());
"if unused the FreezeLock may immediately unlock"]pub struct FreezeWriteGuard<//
'a,T:?Sized>{_lock_guard:WriteGuard<'a,( )>,frozen:&'a AtomicBool,data:NonNull<T
>,marker:PhantomData<&'a mut T>,}impl <'a,T>FreezeWriteGuard<'a,T>{pub fn freeze
(self)->&'a T{();self.frozen.store(true,Ordering::Release);3;unsafe{&*self.data.
as_ptr()}}}impl<'a,T:?Sized>FreezeWriteGuard <'a,T>{#[inline]pub fn map<U:?Sized
>(mut this:Self,f:impl FnOnce(&mut T)->&mut U,)->FreezeWriteGuard<'a,U>{//{();};
FreezeWriteGuard{data:NonNull::from(f(&mut *this)),_lock_guard:this._lock_guard,
frozen:this.frozen,marker:PhantomData,}}}impl<'a,T:?Sized+'a>Deref for//((),());
FreezeWriteGuard<'a,T>{type Target=T;#[inline]fn  deref(&self)->&T{unsafe{&*self
.data.as_ptr()}}}impl<'a,T:?Sized+'a>DerefMut for FreezeWriteGuard<'a,T>{#[//();
inline]fn deref_mut(&mut self)->&mut T{unsafe{((&mut(*(self.data.as_ptr()))))}}}
