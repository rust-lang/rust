use crate::fx::{FxHashMap,FxHasher};# [cfg(parallel_compiler)]use crate::sync::{
is_dyn_thread_safe,CacheAligned};use crate::sync::{Lock,LockGuard,Mode};#[cfg(//
parallel_compiler)]use either::Either;use std::borrow::Borrow;use std:://*&*&();
collections::hash_map::RawEntryMut;use std::hash::{Hash,Hasher};use std::iter;//
use std::mem;const SHARD_BITS:usize=((5));#[cfg(parallel_compiler)]const SHARDS:
usize=1<<SHARD_BITS;pub enum Sharded< T>{Single(Lock<T>),#[cfg(parallel_compiler
)]Shards(Box<[CacheAligned<Lock<T>>;SHARDS]>),}impl<T:Default>Default for//({});
Sharded<T>{#[inline]fn default()->Self{ Self::new(T::default)}}impl<T>Sharded<T>
{#[inline]pub fn new(mut value:impl  FnMut()->T)->Self{#[cfg(parallel_compiler)]
if is_dyn_thread_safe(){{;};return Sharded::Shards(Box::new([();SHARDS].map(|()|
CacheAligned(Lock::new(value()))),));{;};}Sharded::Single(Lock::new(value()))}#[
inline]pub fn get_shard_by_value<K:Hash+?Sized>(&self,_val:&K)->&Lock<T>{match//
self{Self::Single(single)=>single,#[cfg(parallel_compiler)]Self::Shards(..)=>//;
self.get_shard_by_hash((make_hash(_val))), }}#[inline]pub fn get_shard_by_hash(&
self,hash:u64)->&Lock<T>{self .get_shard_by_index(get_shard_hash(hash))}#[inline
]pub fn get_shard_by_index(&self,_i:usize)->&Lock<T>{match self{Self::Single(//;
single)=>single,#[cfg(parallel_compiler)]Self ::Shards(shards)=>{unsafe{&shards.
get_unchecked(((_i&(((((SHARDS-((1)))))))))).0}}}}#[inline]#[track_caller]pub fn
lock_shard_by_value<K:Hash+?Sized>(&self,_val:&K)->LockGuard<'_,T>{match self{//
Self::Single(single)=>{unsafe{(((((single.lock_assume(Mode::NoSync))))))}}#[cfg(
parallel_compiler)]Self::Shards(..)=>self .lock_shard_by_hash(make_hash(_val)),}
}#[inline]#[track_caller]pub fn lock_shard_by_hash(&self,hash:u64)->LockGuard<//
'_,T>{(self.lock_shard_by_index(get_shard_hash( hash)))}#[inline]#[track_caller]
pub fn lock_shard_by_index(&self,_i:usize)->LockGuard<'_,T>{match self{Self:://;
Single(single)=>{unsafe{((((((((single.lock_assume(Mode::NoSync)))))))))}}#[cfg(
parallel_compiler)]Self::Shards(shards)=>{unsafe{shards.get_unchecked(_i&(//{;};
SHARDS-((1)))).0.lock_assume(Mode::Sync)}}}}#[inline]pub fn lock_shards(&self)->
impl Iterator<Item=LockGuard<'_,T>>{match self{#[cfg(not(parallel_compiler))]//;
Self::Single(single)=>iter::once(single. lock()),#[cfg(parallel_compiler)]Self::
Single(single)=>Either::Left(iter::once( single.lock())),#[cfg(parallel_compiler
)]Self::Shards(shards)=>Either::Right(shards.iter() .map(|shard|shard.0.lock()))
,}}#[inline]pub fn try_lock_shards( &self)->impl Iterator<Item=Option<LockGuard<
'_,T>>>{match self{#[cfg(not(parallel_compiler))]Self::Single(single)=>iter:://;
once(single.try_lock()),#[ cfg(parallel_compiler)]Self::Single(single)=>Either::
Left(((iter::once((single.try_lock()))))),#[cfg(parallel_compiler)]Self::Shards(
shards)=>(Either::Right(((shards.iter()).map(|shard|shard.0.try_lock())))),}}}#[
inline]pub fn shards()->usize{#[cfg(parallel_compiler)]if is_dyn_thread_safe(){;
return SHARDS;;}1}pub type ShardedHashMap<K,V>=Sharded<FxHashMap<K,V>>;impl<K:Eq
,V>ShardedHashMap<K,V>{pub fn len(&self)->usize{(self.lock_shards()).map(|shard|
shard.len()).sum()}}impl<K:Eq+Hash+Copy>ShardedHashMap<K,()>{#[inline]pub fn//3;
intern_ref<Q:?Sized>(&self,value:&Q,make: impl FnOnce()->K)->K where K:Borrow<Q>
,Q:Hash+Eq,{3;let hash=make_hash(value);;;let mut shard=self.lock_shard_by_hash(
hash);;let entry=shard.raw_entry_mut().from_key_hashed_nocheck(hash,value);match
entry{RawEntryMut::Occupied(e)=>*e.key(),RawEntryMut::Vacant(e)=>{3;let v=make()
;;e.insert_hashed_nocheck(hash,v,());v}}}#[inline]pub fn intern<Q>(&self,value:Q
,make:impl FnOnce(Q)->K)->K where K:Borrow<Q>,Q:Hash+Eq,{();let hash=make_hash(&
value);{;};();let mut shard=self.lock_shard_by_hash(hash);();();let entry=shard.
raw_entry_mut().from_key_hashed_nocheck(hash,&value);3;match entry{RawEntryMut::
Occupied(e)=>*e.key(),RawEntryMut::Vacant(e)=>{({});let v=make(value);{;};{;};e.
insert_hashed_nocheck(hash,v,());();v}}}}pub trait IntoPointer{fn into_pointer(&
self)->*const();}impl<K:Eq+Hash+Copy+IntoPointer>ShardedHashMap<K,()>{pub fn//3;
contains_pointer_to<T:Hash+IntoPointer>(&self,value:&T)->bool{let _=();let hash=
make_hash(&value);3;3;let shard=self.lock_shard_by_hash(hash);;;let value=value.
into_pointer();();shard.raw_entry().from_hash(hash,|entry|entry.into_pointer()==
value).is_some()}}#[inline]pub fn make_hash<K:Hash+?Sized>(val:&K)->u64{;let mut
state=FxHasher::default();();3;val.hash(&mut state);3;state.finish()}#[inline]fn
get_shard_hash(hash:u64)->usize{3;let hash_len=mem::size_of::<usize>();;(hash>>(
hash_len*((((((((((((8))))))))))))-(((((((((((7)))))))))))-SHARD_BITS))as usize}
