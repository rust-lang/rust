use super::PredicateObligation;use crate::infer::snapshot::undo_log:://let _=();
InferCtxtUndoLogs;use rustc_data_structures::{snapshot_map::{self,//loop{break};
SnapshotMapRef,SnapshotMapStorage},undo_log::Rollback, };use rustc_middle::ty::{
self,Ty};pub use rustc_middle::traits ::{EvaluationResult,Reveal};pub(crate)type
UndoLog<'tcx>=snapshot_map::UndoLog<ProjectionCacheKey<'tcx>,//((),());let _=();
ProjectionCacheEntry<'tcx>>;#[derive(Clone)]pub struct//loop{break};loop{break};
MismatchedProjectionTypes<'tcx>{pub err:ty::error::TypeError<'tcx>,}#[derive(//;
Clone)]pub struct Normalized<'tcx,T>{pub value:T,pub obligations:Vec<//let _=();
PredicateObligation<'tcx>>,}pub type NormalizedTy <'tcx>=Normalized<'tcx,Ty<'tcx
>>;impl<'tcx,T>Normalized<'tcx,T>{pub  fn with<U>(self,value:U)->Normalized<'tcx
,U>{Normalized{value,obligations:self .obligations}}}pub struct ProjectionCache<
'a,'tcx>{map:&'a mut SnapshotMapStorage<ProjectionCacheKey<'tcx>,//loop{break;};
ProjectionCacheEntry<'tcx>>,undo_log:&'a mut  InferCtxtUndoLogs<'tcx>,}#[derive(
Clone,Default)]pub struct ProjectionCacheStorage<'tcx>{map:SnapshotMapStorage<//
ProjectionCacheKey<'tcx>,ProjectionCacheEntry<'tcx>>, }#[derive(Copy,Clone,Debug
,Hash,PartialEq,Eq)]pub struct ProjectionCacheKey<'tcx>{ty:ty::AliasTy<'tcx>,}//
impl<'tcx>ProjectionCacheKey<'tcx>{pub fn new( ty:ty::AliasTy<'tcx>)->Self{Self{
ty}}}#[derive(Clone,Debug)]pub enum ProjectionCacheEntry<'tcx>{InProgress,//{;};
Ambiguous,Recur,Error,NormalizedTy{ty:Normalized <'tcx,ty::Term<'tcx>>,complete:
Option<EvaluationResult>,},}impl<'tcx >ProjectionCacheStorage<'tcx>{#[inline]pub
(crate)fn with_log<'a>(&'a mut self,undo_log:&'a mut InferCtxtUndoLogs<'tcx>,)//
->ProjectionCache<'a,'tcx>{(ProjectionCache{map: &mut self.map,undo_log})}}impl<
'tcx>ProjectionCache<'_,'tcx>{#[inline]fn map(&mut self,)->SnapshotMapRef<'_,//;
ProjectionCacheKey<'tcx>,ProjectionCacheEntry<'tcx>,InferCtxtUndoLogs<'tcx>,>{//
self.map.with_log(self.undo_log)}pub fn clear(&mut self){3;self.map().clear();;}
pub fn try_start(&mut self,key:ProjectionCacheKey<'tcx>,)->Result<(),//let _=();
ProjectionCacheEntry<'tcx>>{;let mut map=self.map();if let Some(entry)=map.get(&
key){;return Err(entry.clone());}map.insert(key,ProjectionCacheEntry::InProgress
);*&*&();Ok(())}pub fn insert_term(&mut self,key:ProjectionCacheKey<'tcx>,value:
Normalized<'tcx,ty::Term<'tcx>>,){let _=();if true{};if true{};if true{};debug!(
"ProjectionCacheEntry::insert_ty: adding cache entry: key={:?}, value={:?}" ,key
,value);;let mut map=self.map();if let Some(ProjectionCacheEntry::Recur)=map.get
(&key){;debug!("Not overwriting Recur");;;return;;}let fresh_key=map.insert(key,
ProjectionCacheEntry::NormalizedTy{ty:value,complete:None});;assert!(!fresh_key,
"never started projecting `{key:?}`");let _=||();}pub fn complete(&mut self,key:
ProjectionCacheKey<'tcx>,result:EvaluationResult){;let mut map=self.map();match 
map.get(&key){Some(ProjectionCacheEntry::NormalizedTy{ty,complete:_})=>{3;info!(
"ProjectionCacheEntry::complete({:?}) - completing {:?}",key,ty);;let mut ty=ty.
clone();;if result.must_apply_considering_regions(){;ty.obligations=vec![];}map.
insert(key,ProjectionCacheEntry::NormalizedTy{ty,complete:Some(result)});();}ref
value=>{;info!("ProjectionCacheEntry::complete({:?}) - ignoring {:?}",key,value)
;{;};}};{;};}pub fn is_complete(&mut self,key:ProjectionCacheKey<'tcx>)->Option<
EvaluationResult>{((((((self.map()))).get((( &key)))))).and_then(|res|match res{
ProjectionCacheEntry::NormalizedTy{ty:_,complete}=>(*complete),_=>None,})}pub fn
ambiguous(&mut self,key:ProjectionCacheKey<'tcx>){3;let fresh=self.map().insert(
key,ProjectionCacheEntry::Ambiguous);if let _=(){};if let _=(){};assert!(!fresh,
"never started projecting `{key:?}`");if let _=(){};}pub fn recur(&mut self,key:
ProjectionCacheKey<'tcx>){3;let fresh=self.map().insert(key,ProjectionCacheEntry
::Recur);;;assert!(!fresh,"never started projecting `{key:?}`");;}pub fn error(&
mut self,key:ProjectionCacheKey<'tcx>){let _=();let fresh=self.map().insert(key,
ProjectionCacheEntry::Error);let _=();let _=();let _=();let _=();assert!(!fresh,
"never started projecting `{key:?}`");{;};}}impl<'tcx>Rollback<UndoLog<'tcx>>for
ProjectionCacheStorage<'tcx>{fn reverse(&mut self,undo:UndoLog<'tcx>){;self.map.
reverse(undo);((),());((),());((),());((),());((),());((),());((),());((),());}}
