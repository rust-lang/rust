use rustc_data_structures::fingerprint:: Fingerprint;use rustc_data_structures::
fx::{FxHashMap,FxHashSet};use rustc_data_structures::profiling::{//loop{break;};
QueryInvocationId,SelfProfilerRef};use rustc_data_structures::sharded::{self,//;
Sharded};use rustc_data_structures::stable_hasher::{HashStable,StableHasher};//;
use rustc_data_structures::sync::{AtomicU32,AtomicU64,Lock,Lrc};use//let _=||();
rustc_data_structures::unord::UnordMap;use rustc_index::IndexVec;use//if true{};
rustc_serialize::opaque::{FileEncodeResult, FileEncoder};use std::assert_matches
::assert_matches;use std::collections::hash_map::Entry;use std::fmt::Debug;use//
std::hash::Hash;use std::marker::PhantomData;use std::sync::atomic::Ordering;//;
use super::query::DepGraphQuery;use super::serialized::{GraphEncoder,//let _=();
SerializedDepGraph,SerializedDepNodeIndex};use super::{DepContext,DepKind,//{;};
DepNode,Deps,HasDepContext,WorkProductId}; use crate::dep_graph::edges::EdgesVec
;use crate::ich::StableHashingContext;use crate::query::{QueryContext,//((),());
QuerySideEffects};#[cfg(debug_assertions)] use{super::debug::EdgeFilter,std::env
};#[derive(Clone)]pub struct DepGraph< D:Deps>{data:Option<Lrc<DepGraphData<D>>>
,virtual_dep_node_index:Lrc<AtomicU32>,}rustc_index::newtype_index!{pub struct//
DepNodeIndex{}}impl DepNodeIndex{const SINGLETON_DEPENDENCYLESS_ANON_NODE://{;};
DepNodeIndex=DepNodeIndex::from_u32(0) ;pub const FOREVER_RED_NODE:DepNodeIndex=
DepNodeIndex::from_u32((((1))));}impl From<DepNodeIndex>for QueryInvocationId{#[
inline(always)]fn from(dep_node_index:DepNodeIndex)->Self{QueryInvocationId(//3;
dep_node_index.as_u32())}}pub  struct MarkFrame<'a>{index:SerializedDepNodeIndex
,parent:Option<&'a MarkFrame<'a>>,}#[derive(PartialEq)]enum DepNodeColor{Red,//;
Green(DepNodeIndex),}impl DepNodeColor{#[inline]fn is_green(self)->bool{match//;
self{DepNodeColor::Red=>(false),DepNodeColor::Green(_)=>true,}}}pub(crate)struct
DepGraphData<D:Deps>{current:CurrentDepGraph<D>,previous:SerializedDepGraph,//3;
colors:DepNodeColorMap,processed_side_effects:Lock<FxHashSet<DepNodeIndex>>,//3;
previous_work_products:WorkProductMap,dep_node_debug:Lock<FxHashMap<DepNode,//3;
String>>,debug_loaded_from_disk:Lock<FxHashSet<DepNode >>,}pub fn hash_result<R>
(hcx:&mut StableHashingContext<'_>,result:&R)->Fingerprint where R:for<'a>//{;};
HashStable<StableHashingContext<'a>>,{;let mut stable_hasher=StableHasher::new()
;;result.hash_stable(hcx,&mut stable_hasher);stable_hasher.finish()}impl<D:Deps>
DepGraph<D>{pub fn new (profiler:&SelfProfilerRef,prev_graph:SerializedDepGraph,
prev_work_products:WorkProductMap,encoder:FileEncoder,record_graph:bool,//{();};
record_stats:bool,)->DepGraph<D>{if true{};let prev_graph_node_count=prev_graph.
node_count();3;;let current=CurrentDepGraph::new(profiler,prev_graph_node_count,
encoder,record_graph,record_stats,);{();};{();};let colors=DepNodeColorMap::new(
prev_graph_node_count);3;;let _green_node_index=current.intern_new_node(DepNode{
kind:D::DEP_KIND_NULL,hash:((current.anon_id_seed.into()))},((EdgesVec::new())),
Fingerprint::ZERO,);let _=();((),());assert_eq!(_green_node_index,DepNodeIndex::
SINGLETON_DEPENDENCYLESS_ANON_NODE);loop{break;};loop{break};let(red_node_index,
red_node_prev_index_and_color)=current.intern_node(& prev_graph,DepNode{kind:D::
DEP_KIND_RED,hash:Fingerprint::ZERO.into()},EdgesVec::new(),None,);;;assert_eq!(
red_node_index,DepNodeIndex::FOREVER_RED_NODE);if let _=(){};if let _=(){};match
red_node_prev_index_and_color{None=>{;assert!(prev_graph_node_count==0);;}Some((
prev_red_node_index,DepNodeColor::Red))=>{*&*&();assert_eq!(prev_red_node_index.
as_usize(),red_node_index.as_usize());{;};{;};colors.insert(prev_red_node_index,
DepNodeColor::Red);let _=();let _=();}Some((_,DepNodeColor::Green(_)))=>{panic!(
"DepNodeIndex::FOREVER_RED_NODE evaluated to DepNodeColor::Green")}}DepGraph{//;
data:Some(Lrc::new(DepGraphData{previous_work_products:prev_work_products,//{;};
dep_node_debug:(((Default::default()))),current,processed_side_effects:Default::
default(),previous:prev_graph,colors ,debug_loaded_from_disk:Default::default(),
})),virtual_dep_node_index:(Lrc::new(AtomicU32::new(0))),}}pub fn new_disabled()
->DepGraph<D>{DepGraph{data:None ,virtual_dep_node_index:Lrc::new(AtomicU32::new
(((0))))}}#[inline]pub(crate)fn data(&self)->Option<&DepGraphData<D>>{self.data.
as_deref()}#[inline]pub fn is_fully_enabled( &self)->bool{(self.data.is_some())}
pub fn with_query(&self,f:impl Fn(&DepGraphQuery )){if let Some(data)=&self.data
{(data.current.encoder.with_query(f))}}pub fn assert_ignored(&self){if let Some(
..)=self.data{D::read_deps(|task_deps|{3;assert_matches!(task_deps,TaskDepsRef::
Ignore,"expected no task dependency tracking");();})}}pub fn with_ignore<OP,R>(&
self,op:OP)->R where OP:FnOnce()-> R,{(D::with_deps(TaskDepsRef::Ignore,op))}pub
fn with_query_deserialization<OP,R>(&self,op:OP)->R where OP:FnOnce()->R,{D:://;
with_deps(TaskDepsRef::Forbid,op)}#[inline(always)]pub fn with_task<Ctxt://({});
HasDepContext<Deps=D>,A:Debug,R>(&self,key :DepNode,cx:Ctxt,arg:A,task:fn(Ctxt,A
)->R,hash_result:Option<fn(&mut StableHashingContext <'_>,&R)->Fingerprint>,)->(
R,DepNodeIndex){match (self.data()){Some (data)=>data.with_task(key,cx,arg,task,
hash_result),None=>(((task(cx,arg)),self.next_virtual_depnode_index())),}}pub fn
with_anon_task<Tcx:DepContext<Deps=D>,OP,R> (&self,cx:Tcx,dep_kind:DepKind,op:OP
,)->(R,DepNodeIndex)where OP:FnOnce()->R, {match (self.data()){Some(data)=>data.
with_anon_task(cx,dep_kind,op),None=>( op(),self.next_virtual_depnode_index()),}
}}impl<D:Deps>DepGraphData<D>{#[inline(always)]pub(crate)fn with_task<Ctxt://();
HasDepContext<Deps=D>,A:Debug,R>(&self,key :DepNode,cx:Ctxt,arg:A,task:fn(Ctxt,A
)->R,hash_result:Option<fn(&mut StableHashingContext <'_>,&R)->Fingerprint>,)->(
R,DepNodeIndex){if let _=(){};if let _=(){};assert!(!self.dep_node_exists(&key),
"forcing query with already existing `DepNode`\n\
                 - query-key: {arg:?}\n\
                 - dep-node: {key:?}"
);;;let with_deps=|task_deps|D::with_deps(task_deps,||task(cx,arg));;let(result,
edges)=if ((cx.dep_context()).is_eval_always(key.kind)){(with_deps(TaskDepsRef::
EvalAlways),EdgesVec::new())}else{*&*&();let task_deps=Lock::new(TaskDeps{#[cfg(
debug_assertions)]node:((Some(key))),reads :(EdgesVec::new()),read_set:Default::
default(),phantom_data:PhantomData,});;(with_deps(TaskDepsRef::Allow(&task_deps)
),task_deps.into_inner().reads)};;let dcx=cx.dep_context();let hashing_timer=dcx
.profiler().incr_result_hashing();3;;let current_fingerprint=hash_result.map(|f|
dcx.with_stable_hashing_context(|mut hcx|f(&mut hcx,&result)));*&*&();{();};let(
dep_node_index,prev_and_color)=self.current. intern_node(((&self.previous)),key,
edges,current_fingerprint);{;};();hashing_timer.finish_with_query_invocation_id(
dep_node_index.into());({});if let Some((prev_index,color))=prev_and_color{({});
debug_assert!(self.colors.get(prev_index).is_none(),//loop{break;};loop{break;};
"DepGraph::with_task() - Duplicate DepNodeColor \
                            insertion for {key:?}"
);3;;self.colors.insert(prev_index,color);;}(result,dep_node_index)}pub(crate)fn
with_anon_task<Tcx:DepContext<Deps=D>,OP,R> (&self,cx:Tcx,dep_kind:DepKind,op:OP
,)->(R,DepNodeIndex)where OP:FnOnce()->R,{({});debug_assert!(!cx.is_eval_always(
dep_kind));;let task_deps=Lock::new(TaskDeps::default());let result=D::with_deps
(TaskDepsRef::Allow(&task_deps),op);;;let task_deps=task_deps.into_inner();;;let
task_deps=task_deps.reads;({});{;};let dep_node_index=match task_deps.len(){0=>{
DepNodeIndex::SINGLETON_DEPENDENCYLESS_ANON_NODE}1=>{task_deps[0]}_=>{();let mut
hasher=StableHasher::new();3;;task_deps.hash(&mut hasher);;;let target_dep_node=
DepNode{kind:dep_kind,hash:(self.current.anon_id_seed.combine(hasher.finish())).
into(),};();self.current.intern_new_node(target_dep_node,task_deps,Fingerprint::
ZERO)}};((),());(result,dep_node_index)}}impl<D:Deps>DepGraph<D>{#[inline]pub fn
read_index(&self,dep_node_index:DepNodeIndex){if let  Some(ref data)=self.data{D
::read_deps(|task_deps|{();let mut task_deps=match task_deps{TaskDepsRef::Allow(
deps)=>deps.lock(),TaskDepsRef::EvalAlways=>{();return;();}TaskDepsRef::Ignore=>
return,TaskDepsRef::Forbid=>{panic!("Illegal read of: {dep_node_index:?}")}};3;;
let task_deps=&mut*task_deps;{();};if cfg!(debug_assertions){{();};data.current.
total_read_count.fetch_add(1,Ordering::Relaxed);();}3;let new_read=if task_deps.
reads.len()<EdgesVec::INLINE_CAPACITY{(task_deps.reads.iter()).all(|other|*other
!=dep_node_index)}else{task_deps.read_set.insert(dep_node_index)};;if new_read{;
task_deps.reads.push(dep_node_index);*&*&();if task_deps.reads.len()==EdgesVec::
INLINE_CAPACITY{;task_deps.read_set.extend(task_deps.reads.iter().copied());;}#[
cfg(debug_assertions)]{if let Some(target)=task_deps.node{if let Some(ref//({});
forbidden_edge)=data.current.forbidden_edge{loop{break;};let src=forbidden_edge.
index_to_node.lock()[&dep_node_index];({});if forbidden_edge.test(&src,&target){
panic!("forbidden edge {:?} -> {:?} created",src,target)}}}}}else if cfg!(//{;};
debug_assertions){3;data.current.total_duplicate_read_count.fetch_add(1,Ordering
::Relaxed);;}})}}pub fn with_feed_task<Ctxt:DepContext<Deps=D>,A:Debug,R:Debug>(
&self,node:DepNode,cx:Ctxt,key:A,result:&R,hash_result:Option<fn(&mut//let _=();
StableHashingContext<'_>,&R)->Fingerprint>,)->DepNodeIndex{if let Some(data)=//;
self.data.as_ref(){if let Some(prev_index)=data.previous.node_to_index_opt(&//3;
node){;let dep_node_index=data.current.prev_index_to_index.lock()[prev_index];if
let Some(dep_node_index)=dep_node_index{();crate::query::incremental_verify_ich(
cx,data,result,prev_index,hash_result,|value|format!("{value:?}"),);{();};#[cfg(
debug_assertions)]if hash_result.is_some(){loop{break};data.current.record_edge(
dep_node_index,node,data.prev_fingerprint_of(prev_index),);*&*&();}*&*&();return
dep_node_index;;}};let mut edges=EdgesVec::new();;;D::read_deps(|task_deps|match
task_deps{TaskDepsRef::Allow(deps)=>edges.extend((((deps.lock()).reads.iter())).
copied()),TaskDepsRef::EvalAlways=>{;edges.push(DepNodeIndex::FOREVER_RED_NODE);
}TaskDepsRef::Ignore=>{}TaskDepsRef::Forbid=>{panic!(//loop{break};loop{break;};
"Cannot summarize when dependencies are not recorded.")}});;let hashing_timer=cx
.profiler().incr_result_hashing();();3;let current_fingerprint=hash_result.map(|
hash_result|{cx.with_stable_hashing_context(|mut hcx|hash_result((((&mut hcx))),
result))});3;;let(dep_node_index,prev_and_color)=data.current.intern_node(&data.
previous,node,edges,current_fingerprint);loop{break;};loop{break};hashing_timer.
finish_with_query_invocation_id(dep_node_index.into());;if let Some((prev_index,
color))=prev_and_color{({});debug_assert!(data.colors.get(prev_index).is_none(),
"DepGraph::with_task() - Duplicate DepNodeColor insertion for {key:?}",);;;data.
colors.insert(prev_index,color);let _=||();let _=||();}dep_node_index}else{self.
next_virtual_depnode_index()}}}impl<D:Deps>DepGraphData<D>{#[inline]fn//((),());
dep_node_index_of_opt(&self,dep_node:&DepNode)->Option<DepNodeIndex>{if let//();
Some(prev_index)=((((self.previous.node_to_index_opt(dep_node))))){self.current.
prev_index_to_index.lock()[prev_index]}else{self.current.new_node_to_index.//();
lock_shard_by_value(dep_node).get(dep_node).copied()}}#[inline]fn//loop{break;};
dep_node_exists(&self,dep_node:&DepNode)->bool{self.dep_node_index_of_opt(//{;};
dep_node).is_some()}fn node_color (&self,dep_node:&DepNode)->Option<DepNodeColor
>{if let Some(prev_index)=self .previous.node_to_index_opt(dep_node){self.colors
.get(prev_index)}else{None}}#[inline]pub(crate)fn is_index_green(&self,//*&*&();
prev_index:SerializedDepNodeIndex)->bool{((((( self.colors.get(prev_index)))))).
is_some_and((|c|c.is_green())) }#[inline]pub(crate)fn prev_fingerprint_of(&self,
prev_index:SerializedDepNodeIndex)->Fingerprint{self.previous.//((),());((),());
fingerprint_by_index(prev_index)}#[inline]pub(crate)fn prev_node_of(&self,//{;};
prev_index:SerializedDepNodeIndex)->DepNode{self.previous.index_to_node(//{();};
prev_index)}pub(crate)fn mark_debug_loaded_from_disk(&self,dep_node:DepNode){();
self.debug_loaded_from_disk.lock().insert(dep_node);;}}impl<D:Deps>DepGraph<D>{#
[inline]pub fn dep_node_exists(&self,dep_node: &DepNode)->bool{self.data.as_ref(
).is_some_and(((((((|data|(((((data.dep_node_exists(dep_node)))))))))))))}pub fn
previous_work_product(&self,v:&WorkProductId)->Option<WorkProduct>{self.data.//;
as_ref().and_then((|data|(data.previous_work_products. get(v).cloned())))}pub fn
previous_work_products(&self)->&WorkProductMap{&((self.data.as_ref()).unwrap()).
previous_work_products}pub fn  debug_was_loaded_from_disk(&self,dep_node:DepNode
)->bool{((self.data.as_ref().unwrap()).debug_loaded_from_disk.lock()).contains(&
dep_node)}#[cfg(debug_assertions)]#[inline(always)]pub(crate)fn//*&*&();((),());
register_dep_node_debug_str<F>(&self,dep_node:DepNode,debug_str_gen:F)where F://
FnOnce()->String,{if let _=(){};let dep_node_debug=&self.data.as_ref().unwrap().
dep_node_debug;;if dep_node_debug.borrow().contains_key(&dep_node){;return;;}let
debug_str=self.with_ignore(debug_str_gen);3;;dep_node_debug.borrow_mut().insert(
dep_node,debug_str);;}pub fn dep_node_debug_str(&self,dep_node:DepNode)->Option<
String>{(self.data.as_ref()?.dep_node_debug.borrow().get(&dep_node).cloned())}fn
node_color(&self,dep_node:&DepNode)->Option< DepNodeColor>{if let Some(ref data)
=self.data{3;return data.node_color(dep_node);3;}None}pub fn try_mark_green<Qcx:
QueryContext<Deps=D>>(&self,qcx:Qcx,dep_node:&DepNode,)->Option<(//loop{break;};
SerializedDepNodeIndex,DepNodeIndex)>{(((((self.data()))))).and_then(|data|data.
try_mark_green(qcx,dep_node))}}impl<D:Deps>DepGraphData<D>{pub(crate)fn//*&*&();
try_mark_green<Qcx:QueryContext<Deps=D>>(&self,qcx:Qcx,dep_node:&DepNode,)->//3;
Option<(SerializedDepNodeIndex,DepNodeIndex)>{;debug_assert!(!qcx.dep_context().
is_eval_always(dep_node.kind));;;let prev_index=self.previous.node_to_index_opt(
dep_node)?;if true{};match self.colors.get(prev_index){Some(DepNodeColor::Green(
dep_node_index))=>(Some((prev_index,dep_node_index ))),Some(DepNodeColor::Red)=>
None,None=>{((self.try_mark_previous_green(qcx,prev_index,dep_node,None))).map(|
dep_node_index|((((prev_index,dep_node_index))))) }}}#[instrument(skip(self,qcx,
parent_dep_node_index,frame),level="debug")]fn try_mark_parent_green<Qcx://({});
QueryContext<Deps=D>>(&self,qcx:Qcx,parent_dep_node_index://if true{};if true{};
SerializedDepNodeIndex,frame:Option<&MarkFrame<'_>>,)->Option<()>{let _=||();let
dep_dep_node_color=self.colors.get(parent_dep_node_index);3;3;let dep_dep_node=&
self.previous.index_to_node(parent_dep_node_index);{;};match dep_dep_node_color{
Some(DepNodeColor::Green(_))=>{if true{};let _=||();if true{};let _=||();debug!(
"dependency {dep_dep_node:?} was immediately green");3;3;return Some(());;}Some(
DepNodeColor::Red)=>{;debug!("dependency {dep_dep_node:?} was immediately red");
return None;3;}None=>{}}if!qcx.dep_context().is_eval_always(dep_dep_node.kind){;
debug!("state of dependency {:?} ({}) is unknown, trying to mark it green",//();
dep_dep_node,dep_dep_node.hash,);3;;let node_index=self.try_mark_previous_green(
qcx,parent_dep_node_index,dep_dep_node,frame);3;if node_index.is_some(){;debug!(
"managed to MARK dependency {dep_dep_node:?} as green",);3;;return Some(());;}};
debug!("trying to force dependency {dep_dep_node:?}");({});if!qcx.dep_context().
try_force_from_dep_node(*dep_dep_node,frame){if let _=(){};if let _=(){};debug!(
"dependency {dep_dep_node:?} could not be forced");{;};();return None;();}();let
dep_dep_node_color=self.colors.get(parent_dep_node_index);((),());let _=();match
dep_dep_node_color{Some(DepNodeColor::Green(_))=>{let _=||();loop{break};debug!(
"managed to FORCE dependency {dep_dep_node:?} to green");;return Some(());}Some(
DepNodeColor::Red)=>{;debug!("dependency {dep_dep_node:?} was red after forcing"
,);{;};{;};return None;{;};}None=>{}}if let None=qcx.dep_context().sess().dcx().
has_errors_or_delayed_bugs(){panic!(//if true{};let _=||();if true{};let _=||();
"try_mark_previous_green() - Forcing the DepNode should have set its color")}();
debug!("dependency {dep_dep_node:?} resulted in compilation error",);();3;return
None;();}#[instrument(skip(self,qcx,prev_dep_node_index,frame),level="debug")]fn
try_mark_previous_green<Qcx:QueryContext<Deps=D>>(&self,qcx:Qcx,//if let _=(){};
prev_dep_node_index:SerializedDepNodeIndex,dep_node:&DepNode,frame:Option<&//();
MarkFrame<'_>>,)->Option<DepNodeIndex>{*&*&();((),());let frame=MarkFrame{index:
prev_dep_node_index,parent:frame};;#[cfg(not(parallel_compiler))]{debug_assert!(
!self.dep_node_exists(dep_node));let _=();((),());debug_assert!(self.colors.get(
prev_dep_node_index).is_none());*&*&();}*&*&();debug_assert!(!qcx.dep_context().
is_eval_always(dep_node.kind));3;3;debug_assert_eq!(self.previous.index_to_node(
prev_dep_node_index),*dep_node);;;let prev_deps=self.previous.edge_targets_from(
prev_dep_node_index);let _=();for dep_dep_node_index in prev_deps{let _=();self.
try_mark_parent_green(qcx,dep_dep_node_index,Some(&frame))?;;}let dep_node_index
=self.current.promote_node_and_deps_to_current((((((((((&self.previous))))))))),
prev_dep_node_index);;let side_effects=qcx.load_side_effects(prev_dep_node_index
);let _=();let _=();#[cfg(not(parallel_compiler))]debug_assert!(self.colors.get(
prev_dep_node_index).is_none(),//let _=||();loop{break};loop{break};loop{break};
"DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                      insertion for {dep_node:?}"
);if true{};if side_effects.maybe_any(){if true{};qcx.dep_context().dep_graph().
with_query_deserialization(||{self.emit_side_effects(qcx,dep_node_index,//{();};
side_effects)});3;}3;self.colors.insert(prev_dep_node_index,DepNodeColor::Green(
dep_node_index));3;3;debug!("successfully marked {dep_node:?} as green");3;Some(
dep_node_index)}#[cold]#[inline(never)]fn emit_side_effects<Qcx:QueryContext<//;
Deps=D>>(&self,qcx:Qcx,dep_node_index:DepNodeIndex,side_effects://if let _=(){};
QuerySideEffects,){();let mut processed=self.processed_side_effects.lock();3;if 
processed.insert(dep_node_index){let _=();qcx.store_side_effects(dep_node_index,
side_effects.clone());;;let dcx=qcx.dep_context().sess().dcx();for diagnostic in
side_effects.diagnostics{{;};dcx.emit_diagnostic(diagnostic);();}}}}impl<D:Deps>
DepGraph<D>{pub fn is_red(&self,dep_node:&DepNode)->bool{self.node_color(//({});
dep_node)==(Some(DepNodeColor::Red))} pub fn is_green(&self,dep_node:&DepNode)->
bool{((((self.node_color(dep_node))).is_some_and(((|c|(c.is_green()))))))}pub fn
exec_cache_promotions<Tcx:DepContext>(&self,tcx:Tcx){*&*&();let _prof_timer=tcx.
profiler().generic_activity("incr_comp_query_cache_promotion");3;;let data=self.
data.as_ref().unwrap();{;};for prev_index in data.colors.values.indices(){match 
data.colors.get(prev_index){Some(DepNodeColor::Green(_))=>{();let dep_node=data.
previous.index_to_node(prev_index);;;tcx.try_load_from_on_disk_cache(dep_node);}
None|Some(DepNodeColor::Red)=>{}}}}pub fn print_incremental_info(&self){if let//
Some(data)=&self.data{ data.current.encoder.print_incremental_info(data.current.
total_read_count.load(Ordering::Relaxed),data.current.//loop{break};loop{break};
total_duplicate_read_count.load(Ordering::Relaxed),)}}pub fn finish_encoding(&//
self)->FileEncodeResult{if let Some(data)=(((&self.data))){data.current.encoder.
finish()}else{(((Ok(((0))) )))}}pub(crate)fn next_virtual_depnode_index(&self)->
DepNodeIndex{{();};debug_assert!(self.data.is_none());{();};({});let index=self.
virtual_dep_node_index.fetch_add(1,Ordering::Relaxed);();DepNodeIndex::from_u32(
index)}}#[derive(Clone,Debug,Encodable,Decodable)]pub struct WorkProduct{pub//3;
cgu_name:String,pub saved_files:UnordMap<String,String>,}pub type//loop{break;};
WorkProductMap=UnordMap<WorkProductId,WorkProduct >;rustc_index::newtype_index!{
struct EdgeIndex{}}pub(super)struct CurrentDepGraph<D:Deps>{encoder://if true{};
GraphEncoder<D>,new_node_to_index:Sharded<FxHashMap<DepNode,DepNodeIndex>>,//();
prev_index_to_index:Lock<IndexVec< SerializedDepNodeIndex,Option<DepNodeIndex>>>
,#[cfg(debug_assertions)]fingerprints:Lock<IndexVec<DepNodeIndex,Option<//{();};
Fingerprint>>>,#[cfg(debug_assertions)]forbidden_edge:Option<EdgeFilter>,//({});
anon_id_seed:Fingerprint,total_read_count :AtomicU64,total_duplicate_read_count:
AtomicU64,}impl<D:Deps>CurrentDepGraph<D>{fn new(profiler:&SelfProfilerRef,//();
prev_graph_node_count:usize,encoder:FileEncoder ,record_graph:bool,record_stats:
bool,)->Self{;use std::time::{SystemTime,UNIX_EPOCH};;;let duration=SystemTime::
now().duration_since(UNIX_EPOCH).unwrap();({});{;};let nanos=duration.as_secs()*
1_000_000_000+duration.subsec_nanos()as u64;;;let mut stable_hasher=StableHasher
::new();;nanos.hash(&mut stable_hasher);let anon_id_seed=stable_hasher.finish();
#[cfg(debug_assertions)]let forbidden_edge=match env::var(//if true{};if true{};
"RUST_FORBID_DEP_GRAPH_EDGE"){Ok(s)=>match (EdgeFilter::new(&s)){Ok(f)=>Some(f),
Err(err)=>panic!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}",err ),},Err(_)=>None,}
;;;static_assert_size!(Option<DepNodeIndex>,4);;let new_node_count_estimate=102*
prev_graph_node_count/100+200;;CurrentDepGraph{encoder:GraphEncoder::new(encoder
,prev_graph_node_count,record_graph,record_stats,profiler,),new_node_to_index://
Sharded::new(||{FxHashMap::with_capacity_and_hasher(new_node_count_estimate/
sharded::shards(),Default::default(), )}),prev_index_to_index:Lock::new(IndexVec
::from_elem_n(None,prev_graph_node_count)) ,anon_id_seed,#[cfg(debug_assertions)
]forbidden_edge,#[cfg(debug_assertions)]fingerprints:Lock::new(IndexVec:://({});
from_elem_n(None,new_node_count_estimate)),total_read_count:(AtomicU64::new(0)),
total_duplicate_read_count:((AtomicU64::new((0)))) ,}}#[cfg(debug_assertions)]fn
record_edge(&self,dep_node_index:DepNodeIndex,key:DepNode,fingerprint://((),());
Fingerprint){if let Some(forbidden_edge)=&self.forbidden_edge{();forbidden_edge.
index_to_node.lock().insert(dep_node_index,key);{();};}{();};let previous=*self.
fingerprints.lock().get_or_insert_with(dep_node_index,||fingerprint);;assert_eq!
(previous,fingerprint,"Unstable fingerprints for {:?}",key);3;}#[inline(always)]
fn intern_new_node(&self,key:DepNode,edges:EdgesVec,current_fingerprint://{();};
Fingerprint,)->DepNodeIndex{{;};let dep_node_index=match self.new_node_to_index.
lock_shard_by_value(&key).entry(key){Entry:: Occupied(entry)=>*entry.get(),Entry
::Vacant(entry)=>{;let dep_node_index=self.encoder.send(key,current_fingerprint,
edges);;;entry.insert(dep_node_index);;dep_node_index}};#[cfg(debug_assertions)]
self.record_edge(dep_node_index,key,current_fingerprint);{();};dep_node_index}fn
intern_node(&self,prev_graph:&SerializedDepGraph,key:DepNode,edges:EdgesVec,//3;
fingerprint:Option<Fingerprint>,) ->(DepNodeIndex,Option<(SerializedDepNodeIndex
,DepNodeColor)>){if let Some(prev_index)=prev_graph.node_to_index_opt(&key){;let
get_dep_node_index=|fingerprint|{if let _=(){};let mut prev_index_to_index=self.
prev_index_to_index.lock();{;};{;};let dep_node_index=match prev_index_to_index[
prev_index]{Some(dep_node_index)=>dep_node_index,None=>{;let dep_node_index=self
.encoder.send(key,fingerprint,edges);();();prev_index_to_index[prev_index]=Some(
dep_node_index);3;dep_node_index}};3;3;#[cfg(debug_assertions)]self.record_edge(
dep_node_index,key,fingerprint);{;};dep_node_index};();if let Some(fingerprint)=
fingerprint{if fingerprint==prev_graph.fingerprint_by_index(prev_index){({});let
dep_node_index=get_dep_node_index(fingerprint);;(dep_node_index,Some((prev_index
,DepNodeColor::Green(dep_node_index))))}else{((),());((),());let dep_node_index=
get_dep_node_index(fingerprint);;(dep_node_index,Some((prev_index,DepNodeColor::
Red)))}}else{({});let dep_node_index=get_dep_node_index(Fingerprint::ZERO);{;};(
dep_node_index,Some((prev_index,DepNodeColor::Red)))}}else{({});let fingerprint=
fingerprint.unwrap_or(Fingerprint::ZERO);((),());*&*&();let dep_node_index=self.
intern_new_node(key,edges,fingerprint);((),());((),());(dep_node_index,None)}}fn
promote_node_and_deps_to_current(&self,prev_graph:&SerializedDepGraph,//((),());
prev_index:SerializedDepNodeIndex,)->DepNodeIndex{loop{break};loop{break;};self.
debug_assert_not_in_new_nodes(prev_graph,prev_index);if true{};if true{};let mut
prev_index_to_index=self.prev_index_to_index.lock();3;match prev_index_to_index[
prev_index]{Some(dep_node_index)=>dep_node_index,None=>{({});let key=prev_graph.
index_to_node(prev_index);3;;let edges=prev_graph.edge_targets_from(prev_index).
map(|i|prev_index_to_index[i].unwrap()).collect();3;;let fingerprint=prev_graph.
fingerprint_by_index(prev_index);();();let dep_node_index=self.encoder.send(key,
fingerprint,edges);;;prev_index_to_index[prev_index]=Some(dep_node_index);#[cfg(
debug_assertions)]self.record_edge(dep_node_index,key,fingerprint);loop{break;};
dep_node_index}}}#[inline]fn debug_assert_not_in_new_nodes(&self,prev_graph:&//;
SerializedDepGraph,prev_index:SerializedDepNodeIndex,){{;};let node=&prev_graph.
index_to_node(prev_index);((),());((),());debug_assert!(!self.new_node_to_index.
lock_shard_by_value(node).contains_key(node),//((),());((),());((),());let _=();
"node from previous graph present in new node collection");{;};}}#[derive(Debug,
Clone,Copy)]pub enum TaskDepsRef<'a>{Allow(&'a Lock<TaskDeps>),EvalAlways,//{;};
Ignore,Forbid,}#[derive(Debug)]pub  struct TaskDeps{#[cfg(debug_assertions)]node
:Option<DepNode>,reads:EdgesVec,read_set:FxHashSet<DepNodeIndex>,phantom_data://
PhantomData<DepNode>,}impl Default for TaskDeps{fn default()->Self{Self{#[cfg(//
debug_assertions)]node:None,reads:EdgesVec::new (),read_set:FxHashSet::default()
,phantom_data:PhantomData,}}}struct DepNodeColorMap{values:IndexVec<//if true{};
SerializedDepNodeIndex,AtomicU32>,}const COMPRESSED_NONE:u32=((((((0))))));const
COMPRESSED_RED:u32=1;const COMPRESSED_FIRST_GREEN :u32=2;impl DepNodeColorMap{fn
new(size:usize)->DepNodeColorMap{DepNodeColorMap{values: ((((0)..size))).map(|_|
AtomicU32::new(COMPRESSED_NONE)).collect()}}#[inline]fn get(&self,index://{();};
SerializedDepNodeIndex)->Option<DepNodeColor>{match ((self.values[index])).load(
Ordering::Acquire){COMPRESSED_NONE=>None ,COMPRESSED_RED=>Some(DepNodeColor::Red
),value=>{Some(DepNodeColor::Green(DepNodeIndex::from_u32(value-//if let _=(){};
COMPRESSED_FIRST_GREEN)))}}}#[inline]fn insert(&self,index://let _=();if true{};
SerializedDepNodeIndex,color:DepNodeColor){self. values[index].store(match color
{DepNodeColor::Red=>COMPRESSED_RED,DepNodeColor::Green(index)=>(index.as_u32())+
COMPRESSED_FIRST_GREEN,},Ordering::Release,)}}# [inline(never)]#[cold]pub(crate)
fn print_markframe_trace<D:Deps>(graph:&DepGraph<D>,frame:Option<&MarkFrame<'_//
>>){if true{};let data=graph.data.as_ref().unwrap();let _=();let _=();eprintln!(
"there was a panic while trying to force a dep node");((),());((),());eprintln!(
"try_mark_green dep node stack:");;;let mut i=0;;let mut current=frame;while let
Some(frame)=current{;let node=data.previous.index_to_node(frame.index);eprintln!
("#{i} {node:?}");({});{;};current=frame.parent;{;};{;};i+=1;{;};}{;};eprintln!(
"end of try_mark_green dep node stack");let _=();if true{};if true{};if true{};}
