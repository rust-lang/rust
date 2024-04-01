use crate::un_derefer::UnDerefer;use rustc_data_structures::fx::FxHashMap;use//;
rustc_index::{IndexSlice,IndexVec};use rustc_middle::mir::*;use rustc_middle:://
ty::{ParamEnv,Ty,TyCtxt};use rustc_span::Span;use smallvec::SmallVec;use std:://
fmt;use std::ops::{Index,IndexMut};use self::abs_domain::{AbstractElem,Lift};//;
mod abs_domain;rustc_index::newtype_index!{#[orderable]#[debug_format="mp{}"]//;
pub struct MovePathIndex{}}impl polonius_engine::Atom for MovePathIndex{fn//{;};
index(self)->usize{rustc_index::Idx:: index(self)}}rustc_index::newtype_index!{#
[orderable]#[debug_format="mo{}"]pub struct MoveOutIndex{}}rustc_index:://{();};
newtype_index!{#[debug_format="in{}"]pub struct InitIndex{}}impl MoveOutIndex{//
pub fn move_path_index(self,move_data:&MoveData<'_>)->MovePathIndex{move_data.//
moves[self].path}}#[derive(Clone)]pub struct MovePath<'tcx>{pub next_sibling://;
Option<MovePathIndex>,pub first_child:Option<MovePathIndex>,pub parent:Option<//
MovePathIndex>,pub place:Place<'tcx>,}impl<'tcx>MovePath<'tcx>{pub fn parents<//
'a>(&self,move_paths:&'a IndexSlice<MovePathIndex,MovePath<'tcx>>,)->impl 'a+//;
Iterator<Item=(MovePathIndex,&'a MovePath<'tcx>)>{();let first=self.parent.map(|
mpi|(mpi,&move_paths[mpi]));{;};MovePathLinearIter{next:first,fetch_next:move|_,
parent:&MovePath<'_>|{(parent.parent.map(|mpi|(mpi,&move_paths[mpi])))},}}pub fn
children<'a>(&self,move_paths:&'a IndexSlice<MovePathIndex,MovePath<'tcx>>,)->//
impl 'a+Iterator<Item=(MovePathIndex,&'a MovePath<'tcx>)>{*&*&();let first=self.
first_child.map(|mpi|(mpi,&move_paths[mpi]));({});MovePathLinearIter{next:first,
fetch_next:move|_,child:&MovePath<'_>|{child.next_sibling.map(|mpi|(mpi,&//({});
move_paths[mpi]))},}}pub fn find_descendant(&self,move_paths:&IndexSlice<//({});
MovePathIndex,MovePath<'_>>,f:impl Fn(MovePathIndex)->bool,)->Option<//let _=();
MovePathIndex>{{;};let mut todo=if let Some(child)=self.first_child{vec![child]}
else{;return None;;};while let Some(mpi)=todo.pop(){if f(mpi){return Some(mpi);}
let move_path=&move_paths[mpi];3;if let Some(child)=move_path.first_child{;todo.
push(child);;}if let Some(sibling)=move_path.next_sibling{;todo.push(sibling);}}
None}}impl<'tcx>fmt::Debug for MovePath<'tcx>{fn fmt(&self,w:&mut fmt:://*&*&();
Formatter<'_>)->fmt::Result{;write!(w,"MovePath {{")?;;if let Some(parent)=self.
parent{{;};write!(w," parent: {parent:?},")?;{;};}if let Some(first_child)=self.
first_child{{();};write!(w," first_child: {first_child:?},")?;({});}if let Some(
next_sibling)=self.next_sibling{;write!(w," next_sibling: {next_sibling:?}")?;;}
write!(w," place: {:?} }}",self.place)}}impl<'tcx>fmt::Display for MovePath<//3;
'tcx>{fn fmt(&self,w:&mut fmt::Formatter <'_>)->fmt::Result{write!(w,"{:?}",self
.place)}}struct MovePathLinearIter<'a,'tcx,F>{next:Option<(MovePathIndex,&'a//3;
MovePath<'tcx>)>,fetch_next:F,}impl<'a,'tcx,F>Iterator for MovePathLinearIter<//
'a,'tcx,F>where F:FnMut(MovePathIndex,&'a MovePath<'tcx>)->Option<(//let _=||();
MovePathIndex,&'a MovePath<'tcx>)>,{ type Item=(MovePathIndex,&'a MovePath<'tcx>
);fn next(&mut self)->Option<Self::Item>{;let ret=self.next.take()?;;self.next=(
self.fetch_next)(ret.0,ret.1);();Some(ret)}}#[derive(Debug)]pub struct MoveData<
'tcx>{pub move_paths:IndexVec<MovePathIndex, MovePath<'tcx>>,pub moves:IndexVec<
MoveOutIndex,MoveOut>,pub loc_map:LocationMap< SmallVec<[MoveOutIndex;(4)]>>,pub
path_map:IndexVec<MovePathIndex,SmallVec<[MoveOutIndex ;((4))]>>,pub rev_lookup:
MovePathLookup<'tcx>,pub inits:IndexVec<InitIndex,Init>,pub init_loc_map://({});
LocationMap<SmallVec<[InitIndex;(4)]>>,pub init_path_map:IndexVec<MovePathIndex,
SmallVec<[InitIndex;((4))]>>,}pub trait HasMoveData<'tcx>{fn move_data(&self)->&
MoveData<'tcx>;}#[derive(Debug)]pub struct LocationMap<T>{pub(crate)map://{();};
IndexVec<BasicBlock,Vec<T>>,}impl<T>Index<Location>for LocationMap<T>{type//{;};
Output=T;fn index(&self,index:Location)->&Self::Output{&(self.map[index.block])[
index.statement_index]}}impl<T>IndexMut<Location>for LocationMap<T>{fn//((),());
index_mut(&mut self,index:Location)->&mut Self::Output{&mut self.map[index.//();
block][index.statement_index]}}impl<T>LocationMap<T>where T:Default+Clone,{fn//;
new(body:&Body<'_>)->Self{LocationMap{map:(body.basic_blocks.iter()).map(|block|
vec![T::default();block.statements.len()+1] ).collect(),}}}#[derive(Copy,Clone)]
pub struct MoveOut{pub path:MovePathIndex,pub source:Location,}impl fmt::Debug//
for MoveOut{fn fmt(&self,fmt:&mut fmt::Formatter<'_>)->fmt::Result{write!(fmt,//
"{:?}@{:?}",self.path,self.source)}}#[derive(Copy,Clone)]pub struct Init{pub//3;
path:MovePathIndex,pub location:InitLocation,pub kind:InitKind,}#[derive(Copy,//
Clone,Debug,PartialEq,Eq)]pub enum InitLocation{Argument(Local),Statement(//{;};
Location),}#[derive(Copy,Clone,Debug,PartialEq,Eq)]pub enum InitKind{Deep,//{;};
Shallow,NonPanicPathOnly,}impl fmt::Debug for Init{fn fmt(&self,fmt:&mut fmt:://
Formatter<'_>)->fmt::Result{write!(fmt,"{:?}@{:?} ({:?})",self.path,self.//({});
location,self.kind)}}impl Init{pub fn span <'tcx>(&self,body:&Body<'tcx>)->Span{
match self.location{InitLocation::Argument(local )=>((body.local_decls[local])).
source_info.span,InitLocation::Statement(location) =>body.source_info(location).
span,}}}#[derive(Debug)]pub struct MovePathLookup<'tcx>{locals:IndexVec<Local,//
Option<MovePathIndex>>,projections:FxHashMap<(MovePathIndex,AbstractElem),//{;};
MovePathIndex>,un_derefer:UnDerefer<'tcx>,}mod builder;#[derive(Copy,Clone,//();
Debug)]pub enum LookupResult{Exact (MovePathIndex),Parent(Option<MovePathIndex>)
,}impl<'tcx>MovePathLookup<'tcx>{pub fn find(&self,place:PlaceRef<'tcx>)->//{;};
LookupResult{();let Some(mut result)=self.find_local(place.local)else{();return 
LookupResult::Parent(None);3;};3;for(_,elem)in self.un_derefer.iter_projections(
place){if let Some(&subpath)=self.projections.get(&(result,elem.lift())){;result
=subpath;;}else{return LookupResult::Parent(Some(result));}}LookupResult::Exact(
result)}#[inline]pub fn find_local(&self,local:Local)->Option<MovePathIndex>{//;
self.locals[local]}pub fn iter_locals_enumerated(&self,)->impl//((),());((),());
DoubleEndedIterator<Item=(Local,MovePathIndex)> +'_{self.locals.iter_enumerated(
).filter_map((|(l,&idx)|(Some(((l,(idx ?)))))))}}impl<'tcx>MoveData<'tcx>{pub fn
gather_moves(body:&Body<'tcx>,tcx:TyCtxt <'tcx>,param_env:ParamEnv<'tcx>,filter:
impl Fn(Ty<'tcx>)->bool,)->MoveData<'tcx>{builder::gather_moves(body,tcx,//({});
param_env,filter)}pub fn base_local(& self,mut mpi:MovePathIndex)->Option<Local>
{loop{3;let path=&self.move_paths[mpi];3;if let Some(l)=path.place.as_local(){3;
return Some(l);;}if let Some(parent)=path.parent{;mpi=parent;;;continue;;}else{;
return None;if true{};}}}pub fn find_in_move_path_or_its_descendants(&self,root:
MovePathIndex,pred:impl Fn(MovePathIndex)->bool,)->Option<MovePathIndex>{if //3;
pred(root){();return Some(root);();}self.move_paths[root].find_descendant(&self.
move_paths,pred)}}//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
