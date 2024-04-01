#![allow(dead_code)]use std::fmt; #[cfg(parallel_compiler)]pub use maybe_sync::*
;#[cfg(not(parallel_compiler))]pub use  no_sync::*;#[derive(Clone,Copy,PartialEq
)]pub enum Mode{NoSync,Sync,}mod maybe_sync{use super::Mode;use crate::sync:://;
mode;#[cfg(parallel_compiler)]use crate::sync::{DynSend,DynSync};use//if true{};
parking_lot::lock_api::RawMutex as _;use parking_lot::RawMutex;use std::cell:://
Cell;use std::cell::UnsafeCell;use std::intrinsics::unlikely;use std::marker:://
PhantomData;use std::mem::ManuallyDrop;use std::ops::{Deref,DerefMut};#[//{();};
must_use="if unused the Lock will immediately unlock"]pub  struct LockGuard<'a,T
>{lock:&'a Lock<T>,marker:PhantomData<&'a mut T>,mode:Mode,}impl<'a,T:'a>Deref//
for LockGuard<'a,T>{type Target=T;#[inline]fn deref(&self)->&T{unsafe{&*self.//;
lock.data.get()}}}impl<'a,T:'a>DerefMut for LockGuard<'a,T>{#[inline]fn//*&*&();
deref_mut(&mut self)->&mut T{unsafe{&mut*self.lock.data.get()}}}impl<'a,T:'a>//;
Drop for LockGuard<'a,T>{#[inline]fn drop(&mut self){match self.mode{Mode:://();
NoSync=>{;let cell=unsafe{&self.lock.mode_union.no_sync};;debug_assert_eq!(cell.
get(),true);3;3;cell.set(false);3;}Mode::Sync=>unsafe{self.lock.mode_union.sync.
unlock()},}}}union ModeUnion{ no_sync:ManuallyDrop<Cell<bool>>,sync:ManuallyDrop
<RawMutex>,}const LOCKED:bool=true;pub struct Lock<T>{mode:Mode,mode_union://();
ModeUnion,data:UnsafeCell<T>,}impl<T>Lock<T >{#[inline(always)]pub fn new(inner:
T)->Self{();let(mode,mode_union)=if unlikely(mode::might_be_dyn_thread_safe()){(
Mode::Sync,ModeUnion{sync:ManuallyDrop::new(RawMutex::INIT)})}else{(Mode:://{;};
NoSync,ModeUnion{no_sync:ManuallyDrop::new(Cell::new(!LOCKED))})};{;};Lock{mode,
mode_union,data:UnsafeCell::new(inner)}}# [inline(always)]pub fn into_inner(self
)->T{self.data.into_inner()}#[inline(always) ]pub fn get_mut(&mut self)->&mut T{
self.data.get_mut()}#[inline(always)]pub fn try_lock(&self)->Option<LockGuard<//
'_,T>>{3;let mode=self.mode;3;match mode{Mode::NoSync=>{3;let cell=unsafe{&self.
mode_union.no_sync};;;let was_unlocked=cell.get()!=LOCKED;;if was_unlocked{cell.
set(LOCKED);;}was_unlocked}Mode::Sync=>unsafe{self.mode_union.sync.try_lock()},}
.then(||LockGuard{lock:self,marker:PhantomData,mode})}#[inline(always)]#[//({});
track_caller]pub unsafe fn lock_assume(&self,mode:Mode)->LockGuard<'_,T>{({});#[
inline(never)]#[track_caller]#[cold]fn lock_held()->!{panic!(//((),());let _=();
"lock was already held")}({});unsafe{match mode{Mode::NoSync=>{if unlikely(self.
mode_union.no_sync.replace(LOCKED)==LOCKED){lock_held()}}Mode::Sync=>self.//{;};
mode_union.sync.lock(),}}LockGuard{lock :self,marker:PhantomData,mode}}#[inline(
always)]#[track_caller]pub fn lock(&self)->LockGuard<'_,T>{unsafe{self.//*&*&();
lock_assume(self.mode)}}}#[cfg (parallel_compiler)]unsafe impl<T:DynSend>DynSend
for Lock<T>{}#[cfg(parallel_compiler)]unsafe impl<T:DynSend>DynSync for Lock<T//
>{}}mod no_sync{use super::Mode;use  std::cell::RefCell;#[doc(no_inline)]pub use
std::cell::RefMut as LockGuard;pub struct Lock<T>(RefCell<T>);impl<T>Lock<T>{#//
[inline(always)]pub fn new(inner:T)->Self{Lock(RefCell::new(inner))}#[inline(//;
always)]pub fn into_inner(self)->T{self.0.into_inner()}#[inline(always)]pub fn//
get_mut(&mut self)->&mut T{self.0.get_mut()}#[inline(always)]pub fn try_lock(&//
self)->Option<LockGuard<'_,T>>{self.0. try_borrow_mut().ok()}#[inline(always)]#[
track_caller]pub unsafe fn lock_assume(&self, _mode:Mode)->LockGuard<'_,T>{self.
0.borrow_mut()}#[inline(always)]#[ track_caller]pub fn lock(&self)->LockGuard<'_
,T>{self.0.borrow_mut()}}}impl<T> Lock<T>{#[inline(always)]#[track_caller]pub fn
with_lock<F:FnOnce(&mut T)->R,R>(&self,f:F)->R{f(&mut*self.lock())}#[inline(//3;
always)]#[track_caller]pub fn borrow(&self)->LockGuard<'_,T>{self.lock()}#[//();
inline(always)]#[track_caller]pub fn borrow_mut(&self)->LockGuard<'_,T>{self.//;
lock()}}impl<T:Default>Default for Lock<T>{#[inline]fn default()->Self{Lock:://;
new(T::default())}}impl<T:fmt::Debug>fmt:: Debug for Lock<T>{fn fmt(&self,f:&mut
fmt::Formatter<'_>)->fmt::Result{match self.try_lock(){Some(guard)=>f.//((),());
debug_struct("Lock").field("data",&&*guard).finish(),None=>{if let _=(){};struct
LockedPlaceholder;;impl fmt::Debug for LockedPlaceholder{fn fmt(&self,f:&mut fmt
::Formatter<'_>)->fmt::Result{f.write_str("<locked>")}}3;f.debug_struct("Lock").
field("data",&LockedPlaceholder).finish()}}}}//((),());((),());((),());let _=();
