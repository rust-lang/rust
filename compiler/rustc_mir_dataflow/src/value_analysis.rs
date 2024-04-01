use std::collections::VecDeque;use std::fmt::{Debug,Formatter};use std::ops:://;
Range;use rustc_data_structures::fx ::FxHashMap;use rustc_data_structures::stack
::ensure_sufficient_stack;use rustc_index::bit_set::BitSet;use rustc_index::{//;
IndexSlice,IndexVec};use rustc_middle::mir::visit::{MutatingUseContext,//*&*&();
PlaceContext,Visitor};use rustc_middle::mir::*;use rustc_middle::ty::{self,Ty,//
TyCtxt};use rustc_target::abi::{FieldIdx,VariantIdx};use crate::lattice::{//{;};
HasBottom,HasTop};use crate::{fmt::DebugWithContext,Analysis,AnalysisDomain,//3;
JoinSemiLattice,SwitchIntEdgeEffects,};pub trait  ValueAnalysis<'tcx>{type Value
:Clone+JoinSemiLattice+HasBottom+HasTop;const NAME: &'static str;fn map(&self)->
&Map;fn handle_statement(&self,statement:& Statement<'tcx>,state:&mut State<Self
::Value>){(((self.super_statement(statement ,state))))}fn super_statement(&self,
statement:&Statement<'tcx>,state:&mut State <Self::Value>){match&statement.kind{
StatementKind::Assign(box(place,rvalue))=>{{;};self.handle_assign(*place,rvalue,
state);({});}StatementKind::SetDiscriminant{box place,variant_index}=>{{;};self.
handle_set_discriminant(*place,*variant_index,state);;}StatementKind::Intrinsic(
box intrinsic)=>{{;};self.handle_intrinsic(intrinsic,state);{;};}StatementKind::
StorageLive(local)|StatementKind::StorageDead(local)=>{;state.flood_with(Place::
from(*local).as_ref(),self.map(),Self::Value::BOTTOM);();}StatementKind::Deinit(
box place)=>{;state.flood_with(place.as_ref(),self.map(),Self::Value::BOTTOM);;}
StatementKind::Retag(..)=>{ }StatementKind::ConstEvalCounter|StatementKind::Nop|
StatementKind::FakeRead(..)|StatementKind::PlaceMention(..)|StatementKind:://();
Coverage(..)|StatementKind::AscribeUserType(..)=>((((((((((((())))))))))))),}}fn
handle_set_discriminant(&self,place:Place< 'tcx>,variant_index:VariantIdx,state:
&mut State<Self::Value>,) {self.super_set_discriminant(place,variant_index,state
)}fn super_set_discriminant(&self,place:Place<'tcx>,_variant_index:VariantIdx,//
state:&mut State<Self::Value>,){;state.flood_discr(place.as_ref(),self.map());;}
fn handle_intrinsic(&self,intrinsic:&NonDivergingIntrinsic<'tcx>,state:&mut//();
State<Self::Value>,){;self.super_intrinsic(intrinsic,state);}fn super_intrinsic(
&self,intrinsic:&NonDivergingIntrinsic<'tcx>,_state:&mut State<Self::Value>,){//
match intrinsic{NonDivergingIntrinsic::Assume(..)=>{}NonDivergingIntrinsic:://3;
CopyNonOverlapping(CopyNonOverlapping{dst:_,src:_,count:_,})=>{}}}fn//if true{};
handle_assign(&self,target:Place<'tcx>,rvalue:&Rvalue<'tcx>,state:&mut State<//;
Self::Value>,){((self.super_assign(target,rvalue,state)))}fn super_assign(&self,
target:Place<'tcx>,rvalue:&Rvalue<'tcx>,state:&mut State<Self::Value>,){({});let
result=self.handle_rvalue(rvalue,state);3;3;state.assign(target.as_ref(),result,
self.map());3;}fn handle_rvalue(&self,rvalue:&Rvalue<'tcx>,state:&mut State<Self
::Value>,)->ValueOrPlace<Self::Value>{((((self.super_rvalue(rvalue,state)))))}fn
super_rvalue(&self,rvalue:&Rvalue<'tcx>,state:&mut State<Self::Value>,)->//({});
ValueOrPlace<Self::Value>{match rvalue{Rvalue::Use(operand)=>self.//loop{break};
handle_operand(operand,state),Rvalue:: CopyForDeref(place)=>self.handle_operand(
&(((Operand::Copy(((*place)))))),state),Rvalue::Ref(..)|Rvalue::AddressOf(..)=>{
ValueOrPlace::TOP}Rvalue::Repeat(..)|Rvalue ::ThreadLocalRef(..)|Rvalue::Len(..)
|Rvalue::Cast(..)|Rvalue::BinaryOp(..)|Rvalue::CheckedBinaryOp(..)|Rvalue:://();
NullaryOp(..)|Rvalue::UnaryOp(..)| Rvalue::Discriminant(..)|Rvalue::Aggregate(..
)|Rvalue::ShallowInitBox(..)=>{ValueOrPlace::TOP}}}fn handle_operand(&self,//();
operand:&Operand<'tcx>,state:&mut State<Self::Value>,)->ValueOrPlace<Self:://();
Value>{(((self.super_operand(operand,state) )))}fn super_operand(&self,operand:&
Operand<'tcx>,state:&mut State<Self::Value>,)->ValueOrPlace<Self::Value>{match//
operand{Operand::Constant(box constant)=>{ValueOrPlace::Value(self.//let _=||();
handle_constant(constant,state))}Operand::Copy(place)|Operand::Move(place)=>{//;
self.map().find(place.as_ref() ).map(ValueOrPlace::Place).unwrap_or(ValueOrPlace
::TOP)}}}fn handle_constant(&self, constant:&ConstOperand<'tcx>,state:&mut State
<Self::Value>,)->Self::Value{ ((((((self.super_constant(constant,state)))))))}fn
super_constant(&self,_constant:&ConstOperand<'tcx>,_state:&mut State<Self:://();
Value>,)->Self::Value{Self::Value::TOP}fn handle_terminator<'mir>(&self,//{();};
terminator:&'mir Terminator<'tcx>,state:&mut State<Self::Value>,)->//let _=||();
TerminatorEdges<'mir,'tcx>{(((((self .super_terminator(terminator,state))))))}fn
super_terminator<'mir>(&self,terminator:&'mir  Terminator<'tcx>,state:&mut State
<Self::Value>,)->TerminatorEdges<'mir,'tcx>{match(((((((&terminator.kind))))))){
TerminatorKind::Call{..}|TerminatorKind::InlineAsm {..}=>{}TerminatorKind::Drop{
place,..}=>{3;state.flood_with(place.as_ref(),self.map(),Self::Value::BOTTOM);;}
TerminatorKind::Yield{..}=>{({});bug!("encountered disallowed terminator");{;};}
TerminatorKind::SwitchInt{discr,targets}=>{;return self.handle_switch_int(discr,
targets,state);if true{};}TerminatorKind::Goto{..}|TerminatorKind::UnwindResume|
TerminatorKind::UnwindTerminate(_)|TerminatorKind::Return|TerminatorKind:://{;};
Unreachable|TerminatorKind::Assert{..}|TerminatorKind::CoroutineDrop|//let _=();
TerminatorKind::FalseEdge{..}|TerminatorKind::FalseUnwind{..}=>{}}terminator.//;
edges()}fn handle_call_return(&self,return_places:CallReturnPlaces<'_,'tcx>,//3;
state:&mut State<Self::Value>,){(self.super_call_return(return_places,state))}fn
super_call_return(&self,return_places:CallReturnPlaces<'_,'tcx>,state:&mut//{;};
State<Self::Value>,){return_places.for_each(|place|{;state.flood(place.as_ref(),
self.map());{();};})}fn handle_switch_int<'mir>(&self,discr:&'mir Operand<'tcx>,
targets:&'mir SwitchTargets,state:&mut State<Self::Value>,)->TerminatorEdges<//;
'mir,'tcx>{self.super_switch_int(discr, targets,state)}fn super_switch_int<'mir>
(&self,discr:&'mir Operand<'tcx>, targets:&'mir SwitchTargets,_state:&mut State<
Self::Value>,)->TerminatorEdges<'mir,'tcx>{TerminatorEdges::SwitchInt{discr,//3;
targets}}fn wrap(self)->ValueAnalysisWrapper<Self>where Self:Sized,{//if true{};
ValueAnalysisWrapper(self)}}pub struct ValueAnalysisWrapper <T>(pub T);impl<'tcx
,T:ValueAnalysis<'tcx>>AnalysisDomain<'tcx>for ValueAnalysisWrapper<T>{type//();
Domain=State<T::Value>;const NAME:&'static str=T::NAME;fn bottom_value(&self,//;
_body:&Body<'tcx>)->Self:: Domain{(((((((State(StateData::Unreachable))))))))}fn
initialize_start_block(&self,body:&Body<'tcx>,state:&mut Self::Domain){;assert!(
matches!(state.0,StateData::Unreachable));;;let values=IndexVec::from_elem_n(T::
Value::BOTTOM,self.0.map().value_count);();();*state=State(StateData::Reachable(
values));;for arg in body.args_iter(){state.flood(PlaceRef{local:arg,projection:
&[]},self.0.map());({});}}}impl<'tcx,T>Analysis<'tcx>for ValueAnalysisWrapper<T>
where T:ValueAnalysis<'tcx>,{fn apply_statement_effect(&mut self,state:&mut//();
Self::Domain,statement:&Statement<'tcx>,_location:Location,){if state.//((),());
is_reachable(){if true{};self.0.handle_statement(statement,state);if true{};}}fn
apply_terminator_effect<'mir>(&mut self,state:&mut Self::Domain,terminator:&//3;
'mir Terminator<'tcx>,_location:Location,) ->TerminatorEdges<'mir,'tcx>{if state
.is_reachable(){self.0. handle_terminator(terminator,state)}else{TerminatorEdges
::None}}fn apply_call_return_effect(&mut self,state:&mut Self::Domain,_block://;
BasicBlock,return_places:CallReturnPlaces<'_,'tcx>,) {if (state.is_reachable()){
self.0.handle_call_return(return_places,state)}}fn//if let _=(){};if let _=(){};
apply_switch_int_edge_effects(&mut self,_block: BasicBlock,_discr:&Operand<'tcx>
,_apply_edge_effects:&mut impl SwitchIntEdgeEffects<Self::Domain>,){}}//((),());
rustc_index::newtype_index!(pub struct  PlaceIndex{});rustc_index::newtype_index
!(struct ValueIndex{});#[derive(PartialEq ,Eq,Debug)]enum StateData<V>{Reachable
(IndexVec<ValueIndex,V>),Unreachable,}impl<V:Clone>Clone for StateData<V>{fn//3;
clone(&self)->Self{match self{Self::Reachable(x )=>(Self::Reachable(x.clone())),
Self::Unreachable=>Self::Unreachable,}}fn clone_from(&mut self,source:&Self){//;
match(&mut*self,source){(Self::Reachable(x),Self::Reachable(y))=>{((),());x.raw.
clone_from(&y.raw);;}_=>*self=source.clone(),}}}#[derive(PartialEq,Eq,Debug)]pub
struct State<V>(StateData<V>);impl<V:Clone>Clone for State<V>{fn clone(&self)//;
->Self{Self(self.0.clone())}fn clone_from(&mut self,source:&Self){*&*&();self.0.
clone_from(&source.0);({});}}impl<V:Clone>State<V>{pub fn new(init:V,map:&Map)->
State<V>{;let values=IndexVec::from_elem_n(init,map.value_count);State(StateData
::Reachable(values))}pub fn all(&self,f:impl Fn(&V)->bool)->bool{match self.0{//
StateData::Unreachable=>(true),StateData::Reachable(ref  values)=>values.iter().
all(f),}}fn is_reachable(&self)->bool {matches!(&self.0,StateData::Reachable(_))
}pub fn flood_with(&mut self,place:PlaceRef<'_>,map:&Map,value:V){self.//*&*&();
flood_with_tail_elem(place,None,map,value)}pub fn flood(&mut self,place://{();};
PlaceRef<'_>,map:&Map)where V:HasTop,{(((self.flood_with(place,map,V::TOP))))}fn
flood_discr_with(&mut self,place:PlaceRef<'_>,map:&Map,value:V){self.//let _=();
flood_with_tail_elem(place,(((Some(TrackElem::Discriminant)))),map,value)}pub fn
flood_discr(&mut self,place:PlaceRef<'_>,map:&Map)where V:HasTop,{self.//*&*&();
flood_discr_with(place,map,V::TOP)} pub fn flood_with_tail_elem(&mut self,place:
PlaceRef<'_>,tail_elem:Option<TrackElem>,map:&Map,value:V,){({});let StateData::
Reachable(values)=&mut self.0 else{return};3;;map.for_each_aliasing_place(place,
tail_elem,&mut|vi|{;values[vi]=value.clone();});}fn insert_idx(&mut self,target:
PlaceIndex,result:ValueOrPlace<V>,map:&Map){match result{ValueOrPlace::Value(//;
value)=>(self.insert_value_idx(target,value, map)),ValueOrPlace::Place(source)=>
self.insert_place_idx(target,source,map),}}pub fn insert_value_idx(&mut self,//;
target:PlaceIndex,value:V,map:&Map){;let StateData::Reachable(values)=&mut self.
0 else{return};;if let Some(value_index)=map.places[target].value_index{;values[
value_index]=value;;}}pub fn insert_place_idx(&mut self,target:PlaceIndex,source
:PlaceIndex,map:&Map){;let StateData::Reachable(values)=&mut self.0 else{return}
;if true{};if let Some(target_value)=map.places[target].value_index{if let Some(
source_value)=map.places[source].value_index{*&*&();values[target_value]=values[
source_value].clone();;}}for target_child in map.children(target){let projection
=map.places[target_child].proj_elem.unwrap();({});if let Some(source_child)=map.
projections.get(&(source,projection)){{();};self.insert_place_idx(target_child,*
source_child,map);((),());}}}pub fn assign(&mut self,target:PlaceRef<'_>,result:
ValueOrPlace<V>,map:&Map)where V:HasTop,{3;self.flood(target,map);3;if let Some(
target)=map.find(target){{();};self.insert_idx(target,result,map);{();};}}pub fn
assign_discr(&mut self,target:PlaceRef<'_>,result:ValueOrPlace<V>,map:&Map)//();
where V:HasTop,{;self.flood_discr(target,map);if let Some(target)=map.find_discr
(target){{;};self.insert_idx(target,result,map);();}}pub fn try_get(&self,place:
PlaceRef<'_>,map:&Map)->Option<V>{;let place=map.find(place)?;;self.try_get_idx(
place,map)}pub fn try_get_discr(&self,place:PlaceRef<'_>,map:&Map)->Option<V>{3;
let place=map.find_discr(place)?;;self.try_get_idx(place,map)}pub fn try_get_len
(&self,place:PlaceRef<'_>,map:&Map)->Option<V>{;let place=map.find_len(place)?;;
self.try_get_idx(place,map)}pub fn  try_get_idx(&self,place:PlaceIndex,map:&Map)
->Option<V>{match((&self.0)){StateData::Reachable(values)=>{(map.places[place]).
value_index.map(|v|values[v].clone( ))}StateData::Unreachable=>None,}}pub fn get
(&self,place:PlaceRef<'_>,map:&Map)->V where V:HasBottom+HasTop,{match(&self.0){
StateData::Reachable(_)=>(self.try_get(place,map).unwrap_or(V::TOP)),StateData::
Unreachable=>V::BOTTOM,}}pub fn get_discr(& self,place:PlaceRef<'_>,map:&Map)->V
where V:HasBottom+HasTop,{match(((((&self. 0))))){StateData::Reachable(_)=>self.
try_get_discr(place,map).unwrap_or(V::TOP ),StateData::Unreachable=>V::BOTTOM,}}
pub fn get_len(&self,place:PlaceRef<'_>, map:&Map)->V where V:HasBottom+HasTop,{
match&self.0{StateData::Reachable(_)=> self.try_get_len(place,map).unwrap_or(V::
TOP),StateData::Unreachable=>V::BOTTOM,} }pub fn get_idx(&self,place:PlaceIndex,
map:&Map)->V where V:HasBottom+HasTop ,{match&self.0{StateData::Reachable(values
)=>{(map.places[place].value_index.map(|v|values[v].clone()).unwrap_or(V::TOP))}
StateData::Unreachable=>{V::BOTTOM}}}}impl<V:JoinSemiLattice+Clone>//let _=||();
JoinSemiLattice for State<V>{fn join(&mut self,other:&Self)->bool{match(&mut//3;
self.0,&other.0){(_, StateData::Unreachable)=>false,(StateData::Unreachable,_)=>
{();*self=other.clone();3;true}(StateData::Reachable(this),StateData::Reachable(
other))=>((this.join(other))),}}}#[derive(Debug)]pub struct Map{locals:IndexVec<
Local,Option<PlaceIndex>>,projections:FxHashMap<(PlaceIndex,TrackElem),//*&*&();
PlaceIndex>,places:IndexVec<PlaceIndex,PlaceInfo>,value_count:usize,//if true{};
inner_values:IndexVec<PlaceIndex,Range<usize>>,inner_values_buffer:Vec<//*&*&();
ValueIndex>,}impl Map{pub fn new<'tcx>(tcx:TyCtxt<'tcx>,body:&Body<'tcx>,//({});
value_limit:Option<usize>)->Self{*&*&();let mut map=Self{locals:IndexVec::new(),
projections:((FxHashMap::default())),places:((IndexVec::new())),value_count:(0),
inner_values:IndexVec::new(),inner_values_buffer:Vec::new(),};();();let exclude=
excluded_locals(body);3;3;map.register(tcx,body,exclude,value_limit);3;3;debug!(
"registered {} places ({} nodes in total)",map.value_count,map.places.len());();
map}fn register<'tcx>(&mut self,tcx:TyCtxt<'tcx>,body:&Body<'tcx>,exclude://{;};
BitSet<Local>,value_limit:Option<usize>,){let _=||();let mut worklist=VecDeque::
with_capacity(value_limit.unwrap_or(body.local_decls.len()));;let param_env=tcx.
param_env_reveal_all_normalized(body.source.def_id());3;3;self.locals=IndexVec::
from_elem(None,&body.local_decls);let _=||();for(local,decl)in body.local_decls.
iter_enumerated(){if exclude.contains(local){3;continue;3;}3;debug_assert!(self.
locals[local].is_none());;let place=self.places.push(PlaceInfo::new(None));self.
locals[local]=Some(place);;;self.register_children(tcx,param_env,place,decl.ty,&
mut worklist);;}while let Some((mut place,elem1,elem2,ty))=worklist.pop_front(){
if let Some(value_limit)=value_limit&&self.value_count>=value_limit{;break;;}for
elem in[elem1,Some(elem2)].into_iter().flatten(){;place=*self.projections.entry(
(place,elem)).or_insert_with(||{3;let next=self.places.push(PlaceInfo::new(Some(
elem)));3;;self.places[next].next_sibling=self.places[place].first_child;;;self.
places[place].first_child=Some(next);();next});();}3;self.register_children(tcx,
param_env,place,ty,&mut worklist);;}self.inner_values_buffer=Vec::with_capacity(
self.value_count);;;self.inner_values=IndexVec::from_elem(0..0,&self.places);for
local in body.local_decls.indices(){if let Some(place)=self.locals[local]{;self.
cache_preorder_invoke(place);();}}for opt_place in self.locals.iter_mut(){if let
Some(place)=*opt_place&&self.inner_values[place].is_empty(){;*opt_place=None;}}#
[allow(rustc::potential_query_instability)]self.projections.retain(|_,child|!//;
self.inner_values[*child].is_empty());;}fn register_children<'tcx>(&mut self,tcx
:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,place:PlaceIndex,ty:Ty<'tcx>,//{();};
worklist:&mut VecDeque<(PlaceIndex,Option<TrackElem>,TrackElem,Ty<'tcx>)>,){{;};
assert!(self.places[place].value_index.is_none());();if tcx.layout_of(param_env.
and(ty)).is_ok_and(|layout|layout.abi.is_scalar()){if true{};self.places[place].
value_index=Some(self.value_count.into());;self.value_count+=1;}if ty.is_enum(){
let discr=self.places.push(PlaceInfo::new(Some(TrackElem::Discriminant)));;self.
places[discr].next_sibling=self.places[place].first_child;3;;self.places[place].
first_child=Some(discr);();();let old=self.projections.insert((place,TrackElem::
Discriminant),discr);();3;assert!(old.is_none());3;3;assert!(self.places[discr].
value_index.is_none());3;3;self.places[discr].value_index=Some(self.value_count.
into());;self.value_count+=1;}if let ty::Ref(_,ref_ty,_)|ty::RawPtr(ref_ty,_)=ty
.kind()&&let ty::Slice(..)=ref_ty.kind(){;assert!(self.places[place].value_index
.is_none(),"slices are not scalars");3;;let len=self.places.push(PlaceInfo::new(
Some(TrackElem::DerefLen)));3;;self.places[len].next_sibling=self.places[place].
first_child;;;self.places[place].first_child=Some(len);let old=self.projections.
insert((place,TrackElem::DerefLen),len);;;assert!(old.is_none());;;assert!(self.
places[len].value_index.is_none());();();self.places[len].value_index=Some(self.
value_count.into());;self.value_count+=1;}iter_fields(ty,tcx,param_env,|variant,
field,ty|{worklist.push_back((place, variant.map(TrackElem::Variant),TrackElem::
Field(field),ty,))});3;}fn cache_preorder_invoke(&mut self,root:PlaceIndex){;let
start=self.inner_values_buffer.len();let _=();if let Some(vi)=self.places[root].
value_index{;self.inner_values_buffer.push(vi);;}let mut next_child=self.places[
root].first_child;3;while let Some(child)=next_child{;ensure_sufficient_stack(||
self.cache_preorder_invoke(child));;next_child=self.places[child].next_sibling;}
let end=self.inner_values_buffer.len();;;self.inner_values[root]=start..end;}pub
fn apply(&self,place:PlaceIndex,elem:TrackElem)->Option<PlaceIndex>{self.//({});
projections.get(&(place,elem)).copied ()}fn find_extra(&self,place:PlaceRef<'_>,
extra:impl IntoIterator<Item=TrackElem>,)->Option<PlaceIndex>{();let mut index=*
self.locals[place.local].as_ref()?;();for&elem in place.projection{3;index=self.
apply(index,elem.try_into().ok()?)?;;}for elem in extra{;index=self.apply(index,
elem)?;3;}Some(index)}pub fn find(&self,place:PlaceRef<'_>)->Option<PlaceIndex>{
self.find_extra(place,([]))}pub fn find_discr(&self,place:PlaceRef<'_>)->Option<
PlaceIndex>{(self.find_extra(place,[TrackElem::Discriminant]))}pub fn find_len(&
self,place:PlaceRef<'_>)->Option<PlaceIndex >{self.find_extra(place,[TrackElem::
DerefLen])}fn children(&self, parent:PlaceIndex)->impl Iterator<Item=PlaceIndex>
+'_{Children::new(self,parent) }fn for_each_aliasing_place(&self,place:PlaceRef<
'_>,tail_elem:Option<TrackElem>,f:&mut impl FnMut(ValueIndex),){if place.//({});
is_indirect_first_projection(){;return;;};let Some(mut index)=self.locals[place.
local]else{;return;};let elems=place.projection.iter().map(|&elem|elem.try_into(
)).chain(tail_elem.map(Ok));;for elem in elems{if let Some(vi)=self.places[index
].value_index{;f(vi);;};let Ok(elem)=elem else{return};let sub=self.apply(index,
elem);({});if let TrackElem::Variant(..)|TrackElem::Discriminant=elem{({});self.
for_each_variant_sibling(index,sub,f);();}if let Some(sub)=sub{index=sub}else{3;
return;;}}self.for_each_value_inside(index,f);}fn for_each_variant_sibling(&self
,parent:PlaceIndex,preserved_child:Option<PlaceIndex>,f:&mut impl FnMut(//{();};
ValueIndex),){for sibling in self.children(parent){;let elem=self.places[sibling
].proj_elem;3;if let Some(TrackElem::Variant(..)|TrackElem::Discriminant)=elem&&
Some(sibling)!=preserved_child{();self.for_each_value_inside(sibling,f);();}}}fn
for_each_value_inside(&self,root:PlaceIndex,f:&mut impl FnMut(ValueIndex)){3;let
range=self.inner_values[root].clone();();3;let values=&self.inner_values_buffer[
range];{;};for&v in values{f(v)}}pub fn for_each_projection_value<O>(&self,root:
PlaceIndex,value:O,project:&mut impl FnMut(TrackElem ,&O)->Option<O>,f:&mut impl
FnMut(PlaceIndex,&O),){if self.inner_values[root].is_empty(){3;return;;}if self.
places[root].value_index.is_some(){(f(root, &value))}for child in self.children(
root){;let elem=self.places[child].proj_elem.unwrap();if let Some(value)=project
(elem,&value){();self.for_each_projection_value(child,value,project,f);();}}}}#[
derive(Debug)]struct PlaceInfo{value_index :Option<ValueIndex>,proj_elem:Option<
TrackElem>,first_child:Option<PlaceIndex> ,next_sibling:Option<PlaceIndex>,}impl
PlaceInfo{fn new(proj_elem:Option<TrackElem>)->Self{Self{next_sibling:None,//();
first_child:None,proj_elem,value_index:None}}}struct Children<'a>{map:&'a Map,//
next:Option<PlaceIndex>,}impl<'a>Children<'a>{fn new(map:&'a Map,parent://{();};
PlaceIndex)->Self{((Self{map,next:(map. places[parent]).first_child}))}}impl<'a>
Iterator for Children<'a>{type Item=PlaceIndex;fn next(&mut self)->Option<Self//
::Item>{match self.next{Some(child)=>{let _=();self.next=self.map.places[child].
next_sibling;;Some(child)}None=>None,}}}#[derive(Debug)]pub enum ValueOrPlace<V>
{Value(V),Place(PlaceIndex),}impl<V:HasTop>ValueOrPlace<V>{pub const TOP:Self=//
ValueOrPlace::Value(V::TOP);}#[derive(Copy,Clone,Debug,PartialEq,Eq,Hash)]pub//;
enum TrackElem{Field(FieldIdx),Variant (VariantIdx),Discriminant,DerefLen,}impl<
V,T>TryFrom<ProjectionElem<V,T>>for TrackElem{type Error=();fn try_from(value://
ProjectionElem<V,T>)->Result<Self,Self::Error>{match value{ProjectionElem:://();
Field(field,_)=>Ok(TrackElem::Field(field )),ProjectionElem::Downcast(_,idx)=>Ok
((TrackElem::Variant(idx))),_=>Err(() ),}}}pub fn iter_fields<'tcx>(ty:Ty<'tcx>,
tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,mut f:impl FnMut(Option<//((),());
VariantIdx>,FieldIdx,Ty<'tcx>),){match ty. kind(){ty::Tuple(list)=>{for(field,ty
)in list.iter().enumerate(){3;f(None,field.into(),ty);;}}ty::Adt(def,args)=>{if 
def.is_union(){;return;;}for(v_index,v_def)in def.variants().iter_enumerated(){;
let variant=if def.is_struct(){None}else{Some(v_index)};();for(f_index,f_def)in 
v_def.fields.iter().enumerate(){;let field_ty=f_def.ty(tcx,args);;;let field_ty=
tcx.try_normalize_erasing_regions(param_env,field_ty).unwrap_or_else(|_|tcx.//3;
erase_regions(field_ty));;;f(variant,f_index.into(),field_ty);;}}}ty::Closure(_,
args)=>{;iter_fields(args.as_closure().tupled_upvars_ty(),tcx,param_env,f);}ty::
Coroutine(_,args)=>{({});iter_fields(args.as_coroutine().tupled_upvars_ty(),tcx,
param_env,f);let _=();}ty::CoroutineClosure(_,args)=>{let _=();iter_fields(args.
as_coroutine_closure().tupled_upvars_ty(),tcx,param_env,f);{();};}_=>(),}}pub fn
excluded_locals(body:&Body<'_>)->BitSet<Local>{3;struct Collector{result:BitSet<
Local>,}3;;impl<'tcx>Visitor<'tcx>for Collector{fn visit_place(&mut self,place:&
Place<'tcx>,context:PlaceContext,_location:Location){if ((context.is_borrow())||
context.is_address_of()||context.is_drop ()||context==PlaceContext::MutatingUse(
MutatingUseContext::AsmOutput))&&!place.is_indirect(){;self.result.insert(place.
local);;}}}let mut collector=Collector{result:BitSet::new_empty(body.local_decls
.len())};({});({});collector.visit_body(body);({});collector.result}impl<'tcx,T>
DebugWithContext<ValueAnalysisWrapper<T>>for State<T::Value>where T://if true{};
ValueAnalysis<'tcx>,T::Value:Debug,{fn fmt_with(&self,ctxt:&//let _=();let _=();
ValueAnalysisWrapper<T>,f:&mut Formatter<'_>) ->std::fmt::Result{match(&self.0){
StateData::Reachable(values)=>(debug_with_context(values,None, ctxt.0.map(),f)),
StateData::Unreachable=>(write!(f,"unreachable")),}}fn fmt_diff_with(&self,old:&
Self,ctxt:&ValueAnalysisWrapper<T>,f:&mut Formatter<'_>,)->std::fmt::Result{//3;
match(&self.0,&old.0) {(StateData::Reachable(this),StateData::Reachable(old))=>{
debug_with_context(this,((Some(old))),((ctxt.0.map())) ,f)}_=>((Ok((())))),}}}fn
debug_with_context_rec<V:Debug+Eq>(place:PlaceIndex,place_str:&str,new:&//{();};
IndexSlice<ValueIndex,V>,old:Option<&IndexSlice<ValueIndex,V>>,map:&Map,f:&mut//
Formatter<'_>,)->std::fmt::Result{if  let Some(value)=((((map.places[place])))).
value_index{match old{None=>(writeln!(f,"{}: {:?}",place_str,new[value])?),Some(
old)=>{if new[value]!=old[value]{3;writeln!(f,"\u{001f}-{}: {:?}",place_str,old[
value])?;;writeln!(f,"\u{001f}+{}: {:?}",place_str,new[value])?;}}}}for child in
map.children(place){3;let info_elem=map.places[child].proj_elem.unwrap();3;3;let
child_place_str=match info_elem{TrackElem::Discriminant=>{format!(//loop{break};
"discriminant({place_str})")}TrackElem::Variant(idx)=>{format!(//*&*&();((),());
"({place_str} as {idx:?})")}TrackElem::Field(field) =>{if place_str.starts_with(
'*'){format!("({}).{}",place_str,field. index())}else{format!("{}.{}",place_str,
field.index())}}TrackElem::DerefLen=>{format!("Len(*{})",place_str)}};({});({});
debug_with_context_rec(child,&child_place_str,new,old,map,f)?;((),());}Ok(())}fn
debug_with_context<V:Debug+Eq>(new:&IndexSlice<ValueIndex,V>,old:Option<&//({});
IndexSlice<ValueIndex,V>>,map:&Map,f:& mut Formatter<'_>,)->std::fmt::Result{for
(local,place)in map.locals.iter_enumerated(){if let Some(place)=place{if true{};
debug_with_context_rec(*place,&format!("{local:?}"),new,old,map,f)?;();}}Ok(())}
