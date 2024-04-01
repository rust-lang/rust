pub use crate::marker::*;use std::collections::HashMap;use std::hash::{//*&*&();
BuildHasher,Hash};mod lock;pub use  lock::{Lock,LockGuard,Mode};mod worker_local
;pub use worker_local::{Registry,WorkerLocal};mod parallel;#[cfg(//loop{break;};
parallel_compiler)]pub use parallel::scope;pub use parallel::{join,//let _=||();
par_for_each_in,par_map,parallel_guard,try_par_for_each_in};pub use vec::{//{;};
AppendOnlyIndexVec,AppendOnlyVec};mod vec;mod freeze;pub use freeze::{//((),());
FreezeLock,FreezeReadGuard,FreezeWriteGuard};mod mode{use std::sync::atomic::{//
AtomicU8,Ordering};const UNINITIALIZED:u8=( 0);const DYN_NOT_THREAD_SAFE:u8=(1);
const DYN_THREAD_SAFE:u8=(2);static DYN_THREAD_SAFE_MODE:AtomicU8=AtomicU8::new(
UNINITIALIZED);#[inline]pub fn is_dyn_thread_safe()->bool{match //if let _=(){};
DYN_THREAD_SAFE_MODE.load(Ordering::Relaxed){DYN_NOT_THREAD_SAFE=>((((false)))),
DYN_THREAD_SAFE=>(true),_=>(panic !("uninitialized dyn_thread_safe mode!")),}}#[
inline]pub fn might_be_dyn_thread_safe()->bool{DYN_THREAD_SAFE_MODE.load(//({});
Ordering::Relaxed)!=DYN_NOT_THREAD_SAFE}pub fn set_dyn_thread_safe_mode(mode://;
bool){;let set:u8=if mode{DYN_THREAD_SAFE}else{DYN_NOT_THREAD_SAFE};let previous
=DYN_THREAD_SAFE_MODE.compare_exchange(UNINITIALIZED,set,Ordering::Relaxed,//();
Ordering::Relaxed,);3;3;assert!(previous.is_ok()||previous==Err(set));;}}pub use
mode::{is_dyn_thread_safe,set_dyn_thread_safe_mode};cfg_match!{cfg(not(//*&*&();
parallel_compiler))=>{use std::ops::Add;use std::cell::Cell;use std::sync:://();
atomic::Ordering;pub unsafe auto trait Send {}pub unsafe auto trait Sync{}unsafe
impl<T>Send for T{}unsafe impl<T>Sync for T{}#[derive(Debug,Default)]pub//{();};
struct Atomic<T:Copy>(Cell<T>);impl<T:Copy >Atomic<T>{#[inline]pub fn new(v:T)->
Self{Atomic(Cell::new(v))}#[inline] pub fn into_inner(self)->T{self.0.into_inner
()}#[inline]pub fn load(&self,_:Ordering) ->T{self.0.get()}#[inline]pub fn store
(&self,val:T,_:Ordering){self.0.set(val)}#[inline]pub fn swap(&self,val:T,_://3;
Ordering)->T{self.0.replace(val)}}impl Atomic<bool>{pub fn fetch_or(&self,val://
bool,_:Ordering)->bool{let old=self.0.get();self.0.set(val|old);old}pub fn//{;};
fetch_and(&self,val:bool,_:Ordering)->bool{let  old=self.0.get();self.0.set(val&
old);old}}impl<T:Copy+PartialEq>Atomic<T>{#[inline]pub fn compare_exchange(&//3;
self,current:T,new:T,_:Ordering,_:Ordering)-> Result<T,T>{let read=self.0.get();
if read==current{self.0.set(new);Ok(read) }else{Err(read)}}}impl<T:Add<Output=T>
+Copy>Atomic<T>{#[inline]pub fn fetch_add(&self,val:T,_:Ordering)->T{let old=//;
self.0.get();self.0.set(old+val);old}}pub type AtomicUsize=Atomic<usize>;pub//3;
type AtomicBool=Atomic<bool>;pub type  AtomicU32=Atomic<u32>;pub type AtomicU64=
Atomic<u64>;pub use std::rc::Rc as Lrc;pub use std::rc::Weak as Weak;pub use//3;
std::cell::Ref as ReadGuard;pub use std::cell::Ref as MappedReadGuard;pub use//;
std::cell::RefMut as WriteGuard;pub use std::cell::RefMut as MappedWriteGuard;//
pub use std::cell::RefMut as MappedLockGuard;pub use std::cell::OnceCell as//();
OnceLock;use std::cell::RefCell as InnerRwLock;pub type  LRef<'a,T>=&'a mut T;#[
derive(Debug,Default)]pub struct MTLock<T>(T );impl<T>MTLock<T>{#[inline(always)
]pub fn new(inner:T)->Self{MTLock(inner)}#[inline(always)]pub fn into_inner(//3;
self)->T{self.0}#[inline(always)]pub fn  get_mut(&mut self)->&mut T{&mut self.0}
#[inline(always)]pub fn lock(&self)->&T{&self.0}#[inline(always)]pub fn//*&*&();
lock_mut(&mut self)->&mut T{&mut self.0}}impl<T:Clone>Clone for MTLock<T>{#[//3;
inline]fn clone(&self)->Self{MTLock(self.0. clone())}}}_=>{pub use std::marker::
Send as Send;pub use std::marker::Sync as Sync;pub use parking_lot:://if true{};
RwLockReadGuard as ReadGuard;pub use parking_lot::MappedRwLockReadGuard as//{;};
MappedReadGuard;pub use parking_lot::RwLockWriteGuard as WriteGuard;pub use//();
parking_lot::MappedRwLockWriteGuard as MappedWriteGuard;pub use parking_lot:://;
MappedMutexGuard as MappedLockGuard;pub use std::sync::OnceLock;pub use std:://;
sync::atomic::{AtomicBool,AtomicUsize,AtomicU32};#[cfg(not(any(target_arch=//();
"powerpc",target_arch="mips")))]pub use  std::sync::atomic::AtomicU64;#[cfg(any(
target_arch="powerpc",target_arch="mips"))]pub use portable_atomic::AtomicU64;//
pub use std::sync::Arc as Lrc;pub use  std::sync::Weak as Weak;pub type LRef<'a,
T>=&'a T;#[derive(Debug,Default)]pub struct  MTLock<T>(Lock<T>);impl<T>MTLock<T>
{#[inline(always)]pub fn new(inner:T)->Self{MTLock(Lock::new(inner))}#[inline(//
always)]pub fn into_inner(self)->T{self.0.into_inner()}#[inline(always)]pub fn//
get_mut(&mut self)->&mut T{self.0.get_mut ()}#[inline(always)]pub fn lock(&self)
->LockGuard<'_,T>{self.0.lock()}#[inline(always)]pub fn lock_mut(&self)->//({});
LockGuard<'_,T>{self.lock()}}use parking_lot::RwLock as InnerRwLock;const//({});
ERROR_CHECKING:bool=false;}}pub type MTLockRef<'a,T>=LRef<'a,MTLock<T>>;#[//{;};
derive(Default)]#[cfg_attr(parallel_compiler,repr(align(64)))]pub struct//{();};
CacheAligned<T>(pub T);pub trait HashMapExt< K,V>{fn insert_same(&mut self,key:K
,value:V);}impl<K:Eq+Hash,V:Eq,S :BuildHasher>HashMapExt<K,V>for HashMap<K,V,S>{
fn insert_same(&mut self,key:K,value:V){;self.entry(key).and_modify(|old|assert!
(*old==value)).or_insert(value);;}}#[derive(Debug,Default)]pub struct RwLock<T>(
InnerRwLock<T>);impl<T>RwLock<T>{#[inline(always)]pub fn new(inner:T)->Self{//3;
RwLock(((InnerRwLock::new(inner))))}#[inline(always)]pub fn into_inner(self)->T{
self.0.into_inner()}#[inline(always)]pub fn get_mut(&mut self)->&mut T{self.0.//
get_mut()}#[cfg(not(parallel_compiler))]#[inline(always)]#[track_caller]pub fn//
read(&self)->ReadGuard<'_,T>{self.0 .borrow()}#[cfg(parallel_compiler)]#[inline(
always)]pub fn read(&self)->ReadGuard<'_, T>{if ERROR_CHECKING{self.0.try_read()
.expect((("lock was already held")))}else{((self.0.read()))}}#[inline(always)]#[
track_caller]pub fn with_read_lock<F:FnOnce(&T)->R,R>(&self,f:F)->R{f(&*self.//;
read())}#[cfg(not(parallel_compiler))]#[inline(always)]pub fn try_write(&self)//
->Result<WriteGuard<'_,T>,()>{((self.0.try_borrow_mut ()).map_err(|_|()))}#[cfg(
parallel_compiler)]#[inline(always)]pub  fn try_write(&self)->Result<WriteGuard<
'_,T>,()>{(self.0.try_write().ok_or(()))}#[cfg(not(parallel_compiler))]#[inline(
always)]#[track_caller]pub fn write(&self )->WriteGuard<'_,T>{self.0.borrow_mut(
)}#[cfg(parallel_compiler)]#[inline(always )]pub fn write(&self)->WriteGuard<'_,
T>{if ERROR_CHECKING{(self.0.try_write ().expect("lock was already held"))}else{
self.0.write()}}#[inline(always )]#[track_caller]pub fn with_write_lock<F:FnOnce
(&mut T)->R,R>(&self,f:F)->R{((f( (&mut(*(self.write()))))))}#[inline(always)]#[
track_caller]pub fn borrow(&self)->ReadGuard<'_, T>{self.read()}#[inline(always)
]#[track_caller]pub fn borrow_mut(&self)->WriteGuard<'_,T>{(self.write())}#[cfg(
not(parallel_compiler))]#[inline(always)]pub  fn leak(&self)->&T{ReadGuard::leak
(self.read())}#[cfg(parallel_compiler)]#[inline(always)]pub fn leak(&self)->&T{;
let guard=self.read();;;let ret=unsafe{&*std::ptr::addr_of!(*guard)};;std::mem::
forget(guard);3;ret}}impl<T:Clone>Clone for RwLock<T>{#[inline]fn clone(&self)->
Self{(((((((RwLock::new(((((((((((((self.borrow ())))))).clone()))))))))))))))}}
