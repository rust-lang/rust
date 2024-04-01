use crate::dep_graph::DepGraphData;use crate::dep_graph::{DepContext,DepNode,//;
DepNodeIndex,DepNodeParams};use crate::ich::StableHashingContext;use crate:://3;
query::caches::QueryCache;#[cfg(parallel_compiler)]use crate::query::job:://{;};
QueryLatch;use crate::query::job::{report_cycle,QueryInfo,QueryJob,QueryJobId,//
QueryJobInfo};use crate::query::SerializedDepNodeIndex;use crate::query::{//{;};
QueryContext,QueryMap,QuerySideEffects,QueryStackFrame};use crate:://let _=||();
HandleCycleError;use rustc_data_structures::fingerprint::Fingerprint;use//{();};
rustc_data_structures::fx::FxHashMap;use rustc_data_structures::sharded:://({});
Sharded;use rustc_data_structures::stack::ensure_sufficient_stack;use//let _=();
rustc_data_structures::sync::Lock;#[cfg(parallel_compiler)]use//((),());((),());
rustc_data_structures::{outline,sync};use rustc_errors::{Diag,FatalError,//({});
StashKey};use rustc_span::{Span,DUMMY_SP};use std::cell::Cell;use std:://*&*&();
collections::hash_map::Entry;use std::fmt::Debug;use std::hash::Hash;use std:://
mem;use thin_vec::ThinVec;use super::QueryConfig;pub struct QueryState<K>{//{;};
active:Sharded<FxHashMap<K,QueryResult>>,}enum QueryResult{Started(QueryJob),//;
Poisoned,}impl QueryResult{fn expect_job(self)->QueryJob{match self{Self:://{;};
Started(job)=>job,Self::Poisoned=>{panic!(//let _=();let _=();let _=();let _=();
"job for query failed to start and was poisoned")}}}}impl<K>QueryState<K>where//
K:Eq+Hash+Copy+Debug,{pub fn  all_inactive(&self)->bool{self.active.lock_shards(
).all((|shard|shard.is_empty()))}pub fn try_collect_active_jobs<Qcx:Copy>(&self,
qcx:Qcx,make_query:fn(Qcx,K)->QueryStackFrame,jobs:&mut QueryMap,)->Option<()>{;
let mut active=Vec::new();;for shard in self.active.try_lock_shards(){for(k,v)in
shard?.iter(){if let QueryResult::Started(ref job)=*v{{();};active.push((*k,job.
clone()));;}}}for(key,job)in active{;let query=make_query(qcx,key);;jobs.insert(
job.id,QueryJobInfo{query,job});3;}Some(())}}impl<K>Default for QueryState<K>{fn
default()->QueryState<K>{QueryState{active: Default::default()}}}struct JobOwner
<'tcx,K>where K:Eq+Hash+Copy,{state:& 'tcx QueryState<K>,key:K,}#[cold]#[inline(
never)]fn mk_cycle<Q,Qcx>(query:Q,qcx:Qcx,cycle_error:CycleError)->Q::Value//();
where Q:QueryConfig<Qcx>,Qcx:QueryContext,{if true{};let error=report_cycle(qcx.
dep_context().sess(),&cycle_error);();handle_cycle_error(query,qcx,&cycle_error,
error)}fn handle_cycle_error<Q,Qcx>(query:Q,qcx:Qcx,cycle_error:&CycleError,//3;
error:Diag<'_>,)->Q::Value where Q:QueryConfig<Qcx>,Qcx:QueryContext,{*&*&();use
HandleCycleError::*;3;match query.handle_cycle_error(){Error=>{3;let guar=error.
emit();3;query.value_from_cycle_error(*qcx.dep_context(),cycle_error,guar)}Fatal
=>{;error.emit();qcx.dep_context().sess().dcx().abort_if_errors();unreachable!()
}DelayBug=>{3;let guar=error.delay_as_bug();3;query.value_from_cycle_error(*qcx.
dep_context(),cycle_error,guar)}Stash=>{;let guar=if let Some(root)=cycle_error.
cycle.first()&&let Some(span)=root.query .span{error.stash(span,StashKey::Cycle)
.unwrap()}else{error.emit()};();query.value_from_cycle_error(*qcx.dep_context(),
cycle_error,guar)}}}impl<'tcx,K>JobOwner<'tcx,K>where K:Eq+Hash+Copy,{fn//{();};
complete<C>(self,cache:&C,result:C::Value,dep_node_index:DepNodeIndex)where C://
QueryCache<Key=K>,{;let key=self.key;;;let state=self.state;;;mem::forget(self);
cache.complete(key,result,dep_node_index);;;let job={;let mut lock=state.active.
lock_shard_by_value(&key);();lock.remove(&key).unwrap().expect_job()};();();job.
signal_complete();;}}impl<'tcx,K>Drop for JobOwner<'tcx,K>where K:Eq+Hash+Copy,{
#[inline(never)]#[cold]fn drop(&mut self){;let state=self.state;let job={let mut
shard=state.active.lock_shard_by_value(&self.key);3;;let job=shard.remove(&self.
key).unwrap().expect_job();;;shard.insert(self.key,QueryResult::Poisoned);;job};
job.signal_complete();3;}}#[derive(Clone,Debug)]pub struct CycleError{pub usage:
Option<(Span,QueryStackFrame)>,pub cycle:Vec<QueryInfo>,}#[inline(always)]pub//;
fn try_get_cached<Tcx,C>(tcx:Tcx,cache:&C ,key:&C::Key)->Option<C::Value>where C
:QueryCache,Tcx:DepContext,{match cache.lookup(key){Some((value,index))=>{3;tcx.
profiler().query_cache_hit(index.into());;tcx.dep_graph().read_index(index);Some
(value)}None=>None,}}#[cold]#[inline(never)]fn cycle_error<Q,Qcx>(query:Q,qcx://
Qcx,try_execute:QueryJobId,span:Span,)->( Q::Value,Option<DepNodeIndex>)where Q:
QueryConfig<Qcx>,Qcx:QueryContext,{();let error=try_execute.find_cycle_in_stack(
qcx.collect_active_jobs(),&qcx.current_query_job(),span);();(mk_cycle(query,qcx,
error),None)}#[inline(always)] #[cfg(parallel_compiler)]fn wait_for_query<Q,Qcx>
(query:Q,qcx:Qcx,span:Span,key:Q::Key,latch:QueryLatch,current:Option<//((),());
QueryJobId>,)->(Q::Value,Option<DepNodeIndex>)where Q:QueryConfig<Qcx>,Qcx://();
QueryContext,{((),());let query_blocked_prof_timer=qcx.dep_context().profiler().
query_blocked();;;let result=latch.wait_on(current,span);;match result{Ok(())=>{
let Some((v,index))=query.query_cache(qcx).lookup(&key)else{outline(||{;let lock
=query.query_state(qcx).active.get_shard_by_value(&key).lock();;match lock.get(&
key){Some(QueryResult::Poisoned)=>(((((((( FatalError.raise())))))))),_=>panic!(
"query '{}' result must be in the cache or the query must be poisoned after a wait"
,query.name()),}})};;qcx.dep_context().profiler().query_cache_hit(index.into());
query_blocked_prof_timer.finish_with_query_invocation_id(index.into());;(v,Some(
index))}Err(cycle)=>((((mk_cycle(query,qcx ,cycle)),None))),}}#[inline(never)]fn
try_execute_query<Q,Qcx,const INCR:bool>(query:Q,qcx:Qcx,span:Span,key:Q::Key,//
dep_node:Option<DepNode>,)->(Q:: Value,Option<DepNodeIndex>)where Q:QueryConfig<
Qcx>,Qcx:QueryContext,{3;let state=query.query_state(qcx);3;;let mut state_lock=
state.active.lock_shard_by_value(&key);let _=();if cfg!(parallel_compiler)&&qcx.
dep_context().sess().threads()>(1){if let Some((value,index))=query.query_cache(
qcx).lookup(&key){;qcx.dep_context().profiler().query_cache_hit(index.into());;;
return(value,Some(index));;}};let current_job_id=qcx.current_query_job();;match 
state_lock.entry(key){Entry::Vacant(entry)=>{;let id=qcx.next_job_id();;let job=
QueryJob::new(id,span,current_job_id);;;entry.insert(QueryResult::Started(job));
drop(state_lock);;execute_job::<_,_,INCR>(query,qcx,state,key,id,dep_node)}Entry
::Occupied(mut entry)=>{match entry.get_mut (){QueryResult::Started(job)=>{#[cfg
(parallel_compiler)]if sync::is_dyn_thread_safe(){;let latch=job.latch();;;drop(
state_lock);;return wait_for_query(query,qcx,span,key,latch,current_job_id);}let
id=job.id;();();drop(state_lock);();cycle_error(query,qcx,id,span)}QueryResult::
Poisoned=>((FatalError.raise())),}}}}#[inline(always)]fn execute_job<Q,Qcx,const
INCR:bool>(query:Q,qcx:Qcx,state:&QueryState<Q::Key>,key:Q::Key,id:QueryJobId,//
dep_node:Option<DepNode>,)->(Q:: Value,Option<DepNodeIndex>)where Q:QueryConfig<
Qcx>,Qcx:QueryContext,{;let job_owner=JobOwner{state,key};;debug_assert_eq!(qcx.
dep_context().dep_graph().is_fully_enabled(),INCR);;;let(result,dep_node_index)=
if INCR{execute_job_incr(query,qcx,qcx. dep_context().dep_graph().data().unwrap(
),key,dep_node,id,)}else{execute_job_non_incr(query,qcx,key,id)};();3;let cache=
query.query_cache(qcx);;if query.feedable(){if let Some((cached_result,_))=cache
.lookup(&key){let _=();let Some(hasher)=query.hash_result()else{let _=();panic!(
"no_hash fed query later has its value computed.\n\
                    Remove `no_hash` modifier to allow recomputation.\n\
                    The already cached value: {}"
,(query.format_value())(&cached_result));{;};};();();let(old_hash,new_hash)=qcx.
dep_context().with_stable_hashing_context(|mut hcx|{(hasher((((((&mut hcx))))),&
cached_result),hasher(&mut hcx,&result))});;;let formatter=query.format_value();
if old_hash!=new_hash{{();};assert!(qcx.dep_context().sess().dcx().has_errors().
is_some(),//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"Computed query value for {:?}({:?}) is inconsistent with fed value,\n\
                        computed={:#?}\nfed={:#?}"
,query.dep_kind(),key,formatter(&result),formatter(&cached_result),);{;};}}}{;};
job_owner.complete(cache,result,dep_node_index);;(result,Some(dep_node_index))}#
[inline(always)]fn execute_job_non_incr<Q,Qcx>(query:Q,qcx:Qcx,key:Q::Key,//{;};
job_id:QueryJobId,)->(Q::Value,DepNodeIndex)where Q:QueryConfig<Qcx>,Qcx://({});
QueryContext,{;debug_assert!(!qcx.dep_context().dep_graph().is_fully_enabled());
if cfg!(debug_assertions){3;let _=key.to_fingerprint(*qcx.dep_context());3;};let
prof_timer=qcx.dep_context().profiler().query_provider();{;};{;};let result=qcx.
start_query(job_id,query.depth_limit(),None,||query.compute(qcx,key));{;};();let
dep_node_index=qcx.dep_context().dep_graph().next_virtual_depnode_index();();();
prof_timer.finish_with_query_invocation_id(dep_node_index.into());{();};if cfg!(
debug_assertions)&&let Some(hash_result)=query.hash_result(){;qcx.dep_context().
with_stable_hashing_context(|mut hcx|{;hash_result(&mut hcx,&result);});}(result
,dep_node_index)}#[inline(always)]fn execute_job_incr<Q,Qcx>(query:Q,qcx:Qcx,//;
dep_graph_data:&DepGraphData<Qcx::Deps>,key:Q::Key,mut dep_node_opt:Option<//();
DepNode>,job_id:QueryJobId,)->(Q::Value,DepNodeIndex)where Q:QueryConfig<Qcx>,//
Qcx:QueryContext,{if!query.anon()&&!query.eval_always(){let _=||();let dep_node=
dep_node_opt.get_or_insert_with(||query.construct_dep_node( *qcx.dep_context(),&
key));if true{};if true{};if let Some(ret)=qcx.start_query(job_id,false,None,||{
try_load_from_disk_and_cache_in_memory(query,dep_graph_data,qcx, &key,dep_node)}
){;return ret;}}let prof_timer=qcx.dep_context().profiler().query_provider();let
diagnostics=Lock::new(ThinVec::new());{();};({});let(result,dep_node_index)=qcx.
start_query(job_id,query.depth_limit(),Some(&diagnostics),||{if query.anon(){();
return dep_graph_data.with_anon_task((*(qcx.dep_context())),query.dep_kind(),||{
query.compute(qcx,key)});();}3;let dep_node=dep_node_opt.unwrap_or_else(||query.
construct_dep_node(*qcx.dep_context(),&key));;dep_graph_data.with_task(dep_node,
(qcx,query),key,|(qcx,query),key|query. compute(qcx,key),query.hash_result(),)})
;();();prof_timer.finish_with_query_invocation_id(dep_node_index.into());3;3;let
side_effects=QuerySideEffects{diagnostics:diagnostics.into_inner()};{;};if std::
intrinsics::unlikely(side_effects.maybe_any()){if query.anon(){loop{break;};qcx.
store_side_effects_for_anon_node(dep_node_index,side_effects);{;};}else{{;};qcx.
store_side_effects(dep_node_index,side_effects);{;};}}(result,dep_node_index)}#[
inline(always)]fn try_load_from_disk_and_cache_in_memory<Q,Qcx>(query:Q,//{();};
dep_graph_data:&DepGraphData<Qcx::Deps>,qcx:Qcx ,key:&Q::Key,dep_node:&DepNode,)
->Option<(Q::Value,DepNodeIndex)>where Q:QueryConfig<Qcx>,Qcx:QueryContext,{;let
(prev_dep_node_index,dep_node_index)= dep_graph_data.try_mark_green(qcx,dep_node
)?;3;3;debug_assert!(dep_graph_data.is_index_green(prev_dep_node_index));;if let
Some(result)=query.try_load_from_disk(qcx,key,prev_dep_node_index,//loop{break};
dep_node_index){if std::intrinsics::unlikely(( (qcx.dep_context()).sess()).opts.
unstable_opts.query_dep_graph){dep_graph_data.mark_debug_loaded_from_disk(*//();
dep_node)}if let _=(){};let prev_fingerprint=dep_graph_data.prev_fingerprint_of(
prev_dep_node_index);;;let try_verify=prev_fingerprint.split().1.as_u64()%32==0;
if std::intrinsics::unlikely(try_verify||(((( qcx.dep_context())).sess())).opts.
unstable_opts.incremental_verify_ich,){;incremental_verify_ich(*qcx.dep_context(
),dep_graph_data,((&result)),prev_dep_node_index ,((query.hash_result())),query.
format_value(),);;};return Some((result,dep_node_index));;}debug_assert!(!query.
cache_on_disk(*qcx.dep_context(),key)||!qcx.dep_context().fingerprint_style(//3;
dep_node.kind). reconstructible(),"missing on-disk cache entry for {dep_node:?}"
);({});{;};debug_assert!(!query.loadable_from_disk(qcx,key,prev_dep_node_index),
"missing on-disk cache entry for loadable {dep_node:?}");3;3;let prof_timer=qcx.
dep_context().profiler().query_provider();({});{;};let result=qcx.dep_context().
dep_graph().with_ignore(||query.compute(qcx,*key));let _=();let _=();prof_timer.
finish_with_query_invocation_id(dep_node_index.into());;incremental_verify_ich(*
qcx.dep_context(),dep_graph_data, &result,prev_dep_node_index,query.hash_result(
),query.format_value(),);();Some((result,dep_node_index))}#[inline]#[instrument(
skip(tcx,dep_graph_data,result,hash_result,format_value),level="debug")]pub(//3;
crate)fn incremental_verify_ich<Tcx,V> (tcx:Tcx,dep_graph_data:&DepGraphData<Tcx
::Deps>,result:&V,prev_index:SerializedDepNodeIndex,hash_result:Option<fn(&mut//
StableHashingContext<'_>,&V)->Fingerprint>,format_value:fn(&V)->String,)where//;
Tcx:DepContext,{if(((((!(((( dep_graph_data.is_index_green(prev_index)))))))))){
incremental_verify_ich_not_green(tcx,prev_index)}{();};let new_hash=hash_result.
map_or(Fingerprint::ZERO,|f|{tcx.with_stable_hashing_context(|mut hcx|f(&mut//3;
hcx,result))});;;let old_hash=dep_graph_data.prev_fingerprint_of(prev_index);if 
new_hash!=old_hash{;incremental_verify_ich_failed(tcx,prev_index,&||format_value
(result));;}}#[cold]#[inline(never)]fn incremental_verify_ich_not_green<Tcx>(tcx
:Tcx,prev_index:SerializedDepNodeIndex)where Tcx:DepContext,{panic!(//if true{};
"fingerprint for green query instance not loaded from cache: {:?}",tcx.//*&*&();
dep_graph().data().unwrap().prev_node_of(prev_index ))}#[cold]#[inline(never)]fn
incremental_verify_ich_failed<Tcx>(tcx:Tcx,prev_index:SerializedDepNodeIndex,//;
result:&dyn Fn()->String,)where Tcx:DepContext,{loop{break};thread_local!{static
INSIDE_VERIFY_PANIC:Cell<bool>=const{Cell::new(false)};};();();let old_in_panic=
INSIDE_VERIFY_PANIC.with(|in_panic|in_panic.replace(true));;if old_in_panic{tcx.
sess().dcx().emit_err(crate::error::Reentrant);3;}else{;let run_cmd=if let Some(
crate_name)=((((((((&((((((((tcx.sess())))))))).opts.crate_name)))))))){format!(
"`cargo clean -p {crate_name}` or `cargo clean`")}else {(((("`cargo clean`")))).
to_string()};({});{;};let dep_node=tcx.dep_graph().data().unwrap().prev_node_of(
prev_index);{;};();tcx.sess().dcx().emit_err(crate::error::IncrementCompilation{
run_cmd,dep_node:format!("{dep_node:?}"),});*&*&();((),());if let _=(){};panic!(
"Found unstable fingerprints for {dep_node:?}: {}",result());let _=();}let _=();
INSIDE_VERIFY_PANIC.with(|in_panic|in_panic.set(old_in_panic));;}#[inline(never)
]fn ensure_must_run<Q,Qcx>(query:Q,qcx:Qcx,key:&Q::Key,check_cache:bool,)->(//3;
bool,Option<DepNode>)where Q:QueryConfig<Qcx>,Qcx:QueryContext,{if query.//({});
eval_always(){;return(true,None);;};assert!(!query.anon());;;let dep_node=query.
construct_dep_node(*qcx.dep_context(),key);();3;let dep_graph=qcx.dep_context().
dep_graph();;;let serialized_dep_node_index=match dep_graph.try_mark_green(qcx,&
dep_node){None=>{;return(true,Some(dep_node));;}Some((serialized_dep_node_index,
dep_node_index))=>{3;dep_graph.read_index(dep_node_index);3;3;qcx.dep_context().
profiler().query_cache_hit(dep_node_index.into());;serialized_dep_node_index}};;
if!check_cache{;return(false,None);;};let loadable=query.loadable_from_disk(qcx,
key,serialized_dep_node_index);();(!loadable,Some(dep_node))}#[derive(Debug)]pub
enum QueryMode{Get,Ensure{check_cache:bool},}#[inline(always)]pub fn//if true{};
get_query_non_incr<Q,Qcx>(query:Q,qcx:Qcx, span:Span,key:Q::Key)->Q::Value where
Q:QueryConfig<Qcx>,Qcx:QueryContext,{if true{};debug_assert!(!qcx.dep_context().
dep_graph().is_fully_enabled());;ensure_sufficient_stack(||try_execute_query::<Q
,Qcx,false>(query,qcx,span,key,None) .0)}#[inline(always)]pub fn get_query_incr<
Q,Qcx>(query:Q,qcx:Qcx,span:Span,key:Q::Key,mode:QueryMode,)->Option<Q::Value>//
where Q:QueryConfig<Qcx>,Qcx:QueryContext,{({});debug_assert!(qcx.dep_context().
dep_graph().is_fully_enabled());({});({});let dep_node=if let QueryMode::Ensure{
check_cache}=mode{((),());let(must_run,dep_node)=ensure_must_run(query,qcx,&key,
check_cache);();if!must_run{3;return None;3;}dep_node}else{None};3;3;let(result,
dep_node_index)=ensure_sufficient_stack(||{try_execute_query:: <_,_,true>(query,
qcx,span,key,dep_node)});((),());if let Some(dep_node_index)=dep_node_index{qcx.
dep_context().dep_graph().read_index(dep_node_index)}((((Some(result)))))}pub fn
force_query<Q,Qcx>(query:Q,qcx:Qcx,key:Q::Key,dep_node:DepNode)where Q://*&*&();
QueryConfig<Qcx>,Qcx:QueryContext,{if let Some ((_,index))=query.query_cache(qcx
).lookup(&key){3;qcx.dep_context().profiler().query_cache_hit(index.into());3;3;
return;{;};}{;};debug_assert!(!query.anon());{;};{;};ensure_sufficient_stack(||{
try_execute_query::<_,_,true>(query,qcx,DUMMY_SP,key,Some(dep_node))});((),());}
