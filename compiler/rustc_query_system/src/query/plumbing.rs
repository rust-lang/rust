//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepKind, DepNode};
use crate::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use crate::query::caches::QueryCache;
use crate::query::config::{QueryDescription, QueryVtable, QueryVtableExt};
use crate::query::job::{QueryInfo, QueryJob, QueryJobId, QueryJobInfo, QueryShardJobId};
use crate::query::{QueryContext, QueryMap};

#[cfg(not(parallel_compiler))]
use rustc_data_structures::cold_path;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHasher};
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::{Lock, LockGuard};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::{Diagnostic, FatalError};
use rustc_span::source_map::DUMMY_SP;
use rustc_span::Span;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};
use std::mem;
use std::num::NonZeroU32;
use std::ptr;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct QueryStateShard<D, Q, K, C> {
    pub(super) cache: C,
    active: FxHashMap<K, QueryResult<D, Q>>,

    /// Used to generate unique ids for active jobs.
    jobs: u32,
}

impl<D, Q, K, C: Default> Default for QueryStateShard<D, Q, K, C> {
    fn default() -> QueryStateShard<D, Q, K, C> {
        QueryStateShard { cache: Default::default(), active: Default::default(), jobs: 0 }
    }
}

pub struct QueryState<D, Q, C: QueryCache> {
    cache: C,
    shards: Sharded<QueryStateShard<D, Q, C::Key, C::Sharded>>,
    #[cfg(debug_assertions)]
    pub cache_hits: AtomicUsize,
}

impl<D, Q, C: QueryCache> QueryState<D, Q, C> {
    #[inline]
    pub(super) fn get_lookup<'tcx>(
        &'tcx self,
        key: &C::Key,
    ) -> QueryLookup<'tcx, D, Q, C::Key, C::Sharded> {
        // We compute the key's hash once and then use it for both the
        // shard lookup and the hashmap lookup. This relies on the fact
        // that both of them use `FxHasher`.
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        let key_hash = hasher.finish();

        let shard = self.shards.get_shard_index_by_hash(key_hash);
        let lock = self.shards.get_shard_by_index(shard).lock();
        QueryLookup { key_hash, shard, lock }
    }
}

/// Indicates the state of a query for a given key in a query map.
enum QueryResult<D, Q> {
    /// An already executing query. The query job can be used to await for its completion.
    Started(QueryJob<D, Q>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

impl<D, Q, C> QueryState<D, Q, C>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    #[inline(always)]
    pub fn iter_results<R>(
        &self,
        f: impl for<'a> FnOnce(
            Box<dyn Iterator<Item = (&'a C::Key, &'a C::Value, DepNodeIndex)> + 'a>,
        ) -> R,
    ) -> R {
        self.cache.iter(&self.shards, |shard| &mut shard.cache, f)
    }

    #[inline(always)]
    pub fn all_inactive(&self) -> bool {
        let shards = self.shards.lock_shards();
        shards.iter().all(|shard| shard.active.is_empty())
    }

    pub fn try_collect_active_jobs(
        &self,
        kind: D,
        make_query: fn(C::Key) -> Q,
        jobs: &mut QueryMap<D, Q>,
    ) -> Option<()> {
        // We use try_lock_shards here since we are called from the
        // deadlock handler, and this shouldn't be locked.
        let shards = self.shards.try_lock_shards()?;
        let shards = shards.iter().enumerate();
        jobs.extend(shards.flat_map(|(shard_id, shard)| {
            shard.active.iter().filter_map(move |(k, v)| {
                if let QueryResult::Started(ref job) = *v {
                    let id = QueryJobId::new(job.id, shard_id, kind);
                    let info = QueryInfo { span: job.span, query: make_query(k.clone()) };
                    Some((id, QueryJobInfo { info, job: job.clone() }))
                } else {
                    None
                }
            })
        }));

        Some(())
    }
}

impl<D, Q, C: QueryCache> Default for QueryState<D, Q, C> {
    fn default() -> QueryState<D, Q, C> {
        QueryState {
            cache: C::default(),
            shards: Default::default(),
            #[cfg(debug_assertions)]
            cache_hits: AtomicUsize::new(0),
        }
    }
}

/// Values used when checking a query cache which can be reused on a cache-miss to execute the query.
pub struct QueryLookup<'tcx, D, Q, K, C> {
    pub(super) key_hash: u64,
    shard: usize,
    pub(super) lock: LockGuard<'tcx, QueryStateShard<D, Q, K, C>>,
}

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
struct JobOwner<'tcx, D, Q, C>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    state: &'tcx QueryState<D, Q, C>,
    key: C::Key,
    id: QueryJobId<D>,
}

impl<'tcx, D, Q, C> JobOwner<'tcx, D, Q, C>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    /// Either gets a `JobOwner` corresponding the query, allowing us to
    /// start executing the query, or returns with the result of the query.
    /// This function assumes that `try_get_cached` is already called and returned `lookup`.
    /// If the query is executing elsewhere, this will wait for it and return the result.
    /// If the query panicked, this will silently panic.
    ///
    /// This function is inlined because that results in a noticeable speed-up
    /// for some compile-time benchmarks.
    #[inline(always)]
    fn try_start<'a, 'b, CTX>(
        tcx: CTX,
        state: &'b QueryState<CTX::DepKind, CTX::Query, C>,
        span: Span,
        key: &C::Key,
        mut lookup: QueryLookup<'a, CTX::DepKind, CTX::Query, C::Key, C::Sharded>,
        query: &QueryVtable<CTX, C::Key, C::Value>,
    ) -> TryGetJob<'b, CTX::DepKind, CTX::Query, C>
    where
        CTX: QueryContext,
    {
        let lock = &mut *lookup.lock;

        let (latch, mut _query_blocked_prof_timer) = match lock.active.entry((*key).clone()) {
            Entry::Occupied(mut entry) => {
                match entry.get_mut() {
                    QueryResult::Started(job) => {
                        // For parallel queries, we'll block and wait until the query running
                        // in another thread has completed. Record how long we wait in the
                        // self-profiler.
                        let _query_blocked_prof_timer = if cfg!(parallel_compiler) {
                            Some(tcx.profiler().query_blocked())
                        } else {
                            None
                        };

                        // Create the id of the job we're waiting for
                        let id = QueryJobId::new(job.id, lookup.shard, query.dep_kind);

                        (job.latch(id), _query_blocked_prof_timer)
                    }
                    QueryResult::Poisoned => FatalError.raise(),
                }
            }
            Entry::Vacant(entry) => {
                // No job entry for this query. Return a new one to be started later.

                // Generate an id unique within this shard.
                let id = lock.jobs.checked_add(1).unwrap();
                lock.jobs = id;
                let id = QueryShardJobId(NonZeroU32::new(id).unwrap());

                let global_id = QueryJobId::new(id, lookup.shard, query.dep_kind);

                let job = tcx.current_query_job();
                let job = QueryJob::new(id, span, job);

                entry.insert(QueryResult::Started(job));

                let owner = JobOwner { state, id: global_id, key: (*key).clone() };
                return TryGetJob::NotYetStarted(owner);
            }
        };
        mem::drop(lookup.lock);

        // If we are single-threaded we know that we have cycle error,
        // so we just return the error.
        #[cfg(not(parallel_compiler))]
        return TryGetJob::Cycle(cold_path(|| {
            let error: CycleError<CTX::Query> = latch.find_cycle_in_stack(
                tcx.try_collect_active_jobs().unwrap(),
                &tcx.current_query_job(),
                span,
            );
            let value = query.handle_cycle_error(tcx, error);
            state.cache.store_nocache(value)
        }));

        // With parallel queries we might just have to wait on some other
        // thread.
        #[cfg(parallel_compiler)]
        {
            let result = latch.wait_on(tcx.current_query_job(), span);

            if let Err(cycle) = result {
                let value = query.handle_cycle_error(tcx, cycle);
                let value = state.cache.store_nocache(value);
                return TryGetJob::Cycle(value);
            }

            let cached = try_get_cached(
                tcx,
                state,
                (*key).clone(),
                |value, index| (value.clone(), index),
                |_, _| panic!("value must be in cache after waiting"),
            );

            if let Some(prof_timer) = _query_blocked_prof_timer.take() {
                prof_timer.finish_with_query_invocation_id(cached.1.into());
            }

            return TryGetJob::JobCompleted(cached);
        }
    }

    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter and forgets the JobOwner, so it won't poison the query
    #[inline(always)]
    fn complete(self, result: C::Value, dep_node_index: DepNodeIndex) -> C::Stored {
        // We can move out of `self` here because we `mem::forget` it below
        let key = unsafe { ptr::read(&self.key) };
        let state = self.state;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

        let (job, result) = {
            let mut lock = state.shards.get_shard_by_value(&key).lock();
            let job = match lock.active.remove(&key).unwrap() {
                QueryResult::Started(job) => job,
                QueryResult::Poisoned => panic!(),
            };
            let result = state.cache.complete(&mut lock.cache, key, result, dep_node_index);
            (job, result)
        };

        job.signal_complete();
        result
    }
}

#[inline(always)]
fn with_diagnostics<F, R>(f: F) -> (R, ThinVec<Diagnostic>)
where
    F: FnOnce(Option<&Lock<ThinVec<Diagnostic>>>) -> R,
{
    let diagnostics = Lock::new(ThinVec::new());
    let result = f(Some(&diagnostics));
    (result, diagnostics.into_inner())
}

impl<'tcx, D, Q, C> Drop for JobOwner<'tcx, D, Q, C>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        // Poison the query so jobs waiting on it panic.
        let state = self.state;
        let shard = state.shards.get_shard_by_value(&self.key);
        let job = {
            let mut shard = shard.lock();
            let job = match shard.active.remove(&self.key).unwrap() {
                QueryResult::Started(job) => job,
                QueryResult::Poisoned => panic!(),
            };
            shard.active.insert(self.key.clone(), QueryResult::Poisoned);
            job
        };
        // Also signal the completion of the job, so waiters
        // will continue execution.
        job.signal_complete();
    }
}

#[derive(Clone)]
pub struct CycleError<Q> {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, Q)>,
    pub cycle: Vec<QueryInfo<Q>>,
}

/// The result of `try_start`.
enum TryGetJob<'tcx, D, Q, C>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    C: QueryCache,
{
    /// The query is not yet started. Contains a guard to the cache eventually used to start it.
    NotYetStarted(JobOwner<'tcx, D, Q, C>),

    /// The query was already completed.
    /// Returns the result of the query and its dep-node index
    /// if it succeeded or a cycle error if it failed.
    #[cfg(parallel_compiler)]
    JobCompleted((C::Stored, DepNodeIndex)),

    /// Trying to execute the query resulted in a cycle.
    Cycle(C::Stored),
}

/// Checks if the query is already computed and in the cache.
/// It returns the shard index and a lock guard to the shard,
/// which will be used if the query is not in the cache and we need
/// to compute it.
#[inline(always)]
fn try_get_cached<CTX, C, R, OnHit, OnMiss>(
    tcx: CTX,
    state: &QueryState<CTX::DepKind, CTX::Query, C>,
    key: C::Key,
    // `on_hit` can be called while holding a lock to the query cache
    on_hit: OnHit,
    on_miss: OnMiss,
) -> R
where
    C: QueryCache,
    CTX: QueryContext,
    OnHit: FnOnce(&C::Stored, DepNodeIndex) -> R,
    OnMiss: FnOnce(C::Key, QueryLookup<'_, CTX::DepKind, CTX::Query, C::Key, C::Sharded>) -> R,
{
    state.cache.lookup(
        state,
        key,
        |value, index| {
            if unlikely!(tcx.profiler().enabled()) {
                tcx.profiler().query_cache_hit(index.into());
            }
            #[cfg(debug_assertions)]
            {
                state.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            on_hit(value, index)
        },
        on_miss,
    )
}

#[inline(always)]
fn try_execute_query<CTX, C>(
    tcx: CTX,
    state: &QueryState<CTX::DepKind, CTX::Query, C>,
    span: Span,
    key: C::Key,
    lookup: QueryLookup<'_, CTX::DepKind, CTX::Query, C::Key, C::Sharded>,
    query: &QueryVtable<CTX, C::Key, C::Value>,
) -> C::Stored
where
    C: QueryCache,
    C::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    let job = match JobOwner::<'_, CTX::DepKind, CTX::Query, C>::try_start(
        tcx, state, span, &key, lookup, query,
    ) {
        TryGetJob::NotYetStarted(job) => job,
        TryGetJob::Cycle(result) => return result,
        #[cfg(parallel_compiler)]
        TryGetJob::JobCompleted((v, index)) => {
            tcx.dep_graph().read_index(index);
            return v;
        }
    };

    // Fast path for when incr. comp. is off. `to_dep_node` is
    // expensive for some `DepKind`s.
    if !tcx.dep_graph().is_fully_enabled() {
        let null_dep_node = DepNode::new_no_params(DepKind::NULL);
        return force_query_with_job(tcx, key, job, null_dep_node, query).0;
    }

    if query.anon {
        let prof_timer = tcx.profiler().query_provider();

        let ((result, dep_node_index), diagnostics) = with_diagnostics(|diagnostics| {
            tcx.start_query(job.id, diagnostics, |tcx| {
                tcx.dep_graph().with_anon_task(query.dep_kind, || query.compute(tcx, key))
            })
        });

        prof_timer.finish_with_query_invocation_id(dep_node_index.into());

        tcx.dep_graph().read_index(dep_node_index);

        if unlikely!(!diagnostics.is_empty()) {
            tcx.store_diagnostics_for_anon_node(dep_node_index, diagnostics);
        }

        return job.complete(result, dep_node_index);
    }

    let dep_node = query.to_dep_node(tcx, &key);

    if !query.eval_always {
        // The diagnostics for this query will be
        // promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        let loaded = tcx.start_query(job.id, None, |tcx| {
            let marked = tcx.dep_graph().try_mark_green_and_read(tcx, &dep_node);
            marked.map(|(prev_dep_node_index, dep_node_index)| {
                (
                    load_from_disk_and_cache_in_memory(
                        tcx,
                        key.clone(),
                        prev_dep_node_index,
                        dep_node_index,
                        &dep_node,
                        query,
                    ),
                    dep_node_index,
                )
            })
        });
        if let Some((result, dep_node_index)) = loaded {
            return job.complete(result, dep_node_index);
        }
    }

    let (result, dep_node_index) = force_query_with_job(tcx, key, job, dep_node, query);
    tcx.dep_graph().read_index(dep_node_index);
    result
}

fn load_from_disk_and_cache_in_memory<CTX, K, V>(
    tcx: CTX,
    key: K,
    prev_dep_node_index: SerializedDepNodeIndex,
    dep_node_index: DepNodeIndex,
    dep_node: &DepNode<CTX::DepKind>,
    query: &QueryVtable<CTX, K, V>,
) -> V
where
    CTX: QueryContext,
{
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    debug_assert!(tcx.dep_graph().is_green(dep_node));

    // First we try to load the result from the on-disk cache.
    let result = if query.cache_on_disk(tcx, &key, None) {
        let prof_timer = tcx.profiler().incr_cache_loading();
        let result = query.try_load_from_disk(tcx, prev_dep_node_index);
        prof_timer.finish_with_query_invocation_id(dep_node_index.into());

        // We always expect to find a cached result for things that
        // can be forced from `DepNode`.
        debug_assert!(
            !dep_node.kind.can_reconstruct_query_key() || result.is_some(),
            "missing on-disk cache entry for {:?}",
            dep_node
        );
        result
    } else {
        // Some things are never cached on disk.
        None
    };

    let result = if let Some(result) = result {
        result
    } else {
        // We could not load a result from the on-disk cache, so
        // recompute.
        let prof_timer = tcx.profiler().query_provider();

        // The dep-graph for this computation is already in-place.
        let result = tcx.dep_graph().with_ignore(|| query.compute(tcx, key));

        prof_timer.finish_with_query_invocation_id(dep_node_index.into());

        result
    };

    // If `-Zincremental-verify-ich` is specified, re-hash results from
    // the cache and make sure that they have the expected fingerprint.
    if unlikely!(tcx.incremental_verify_ich()) {
        incremental_verify_ich(tcx, &result, dep_node, dep_node_index, query);
    }

    result
}

#[inline(never)]
#[cold]
fn incremental_verify_ich<CTX, K, V>(
    tcx: CTX,
    result: &V,
    dep_node: &DepNode<CTX::DepKind>,
    dep_node_index: DepNodeIndex,
    query: &QueryVtable<CTX, K, V>,
) where
    CTX: QueryContext,
{
    assert!(
        Some(tcx.dep_graph().fingerprint_of(dep_node_index))
            == tcx.dep_graph().prev_fingerprint_of(dep_node),
        "fingerprint for green query instance not loaded from cache: {:?}",
        dep_node,
    );

    debug!("BEGIN verify_ich({:?})", dep_node);
    let mut hcx = tcx.create_stable_hashing_context();

    let new_hash = query.hash_result(&mut hcx, result).unwrap_or(Fingerprint::ZERO);
    debug!("END verify_ich({:?})", dep_node);

    let old_hash = tcx.dep_graph().fingerprint_of(dep_node_index);

    assert!(new_hash == old_hash, "found unstable fingerprints for {:?}", dep_node,);
}

#[inline(always)]
fn force_query_with_job<C, CTX>(
    tcx: CTX,
    key: C::Key,
    job: JobOwner<'_, CTX::DepKind, CTX::Query, C>,
    dep_node: DepNode<CTX::DepKind>,
    query: &QueryVtable<CTX, C::Key, C::Value>,
) -> (C::Stored, DepNodeIndex)
where
    C: QueryCache,
    CTX: QueryContext,
{
    // If the following assertion triggers, it can have two reasons:
    // 1. Something is wrong with DepNode creation, either here or
    //    in `DepGraph::try_mark_green()`.
    // 2. Two distinct query keys get mapped to the same `DepNode`
    //    (see for example #48923).
    assert!(
        !tcx.dep_graph().dep_node_exists(&dep_node),
        "forcing query with already existing `DepNode`\n\
                 - query-key: {:?}\n\
                 - dep-node: {:?}",
        key,
        dep_node
    );

    let prof_timer = tcx.profiler().query_provider();

    let ((result, dep_node_index), diagnostics) = with_diagnostics(|diagnostics| {
        tcx.start_query(job.id, diagnostics, |tcx| {
            if query.eval_always {
                tcx.dep_graph().with_eval_always_task(
                    dep_node,
                    tcx,
                    key,
                    query.compute,
                    query.hash_result,
                )
            } else {
                tcx.dep_graph().with_task(dep_node, tcx, key, query.compute, query.hash_result)
            }
        })
    });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    if unlikely!(!diagnostics.is_empty()) && dep_node.kind != DepKind::NULL {
        tcx.store_diagnostics(dep_node_index, diagnostics);
    }

    let result = job.complete(result, dep_node_index);

    (result, dep_node_index)
}

#[inline(never)]
fn get_query_impl<CTX, C>(
    tcx: CTX,
    state: &QueryState<CTX::DepKind, CTX::Query, C>,
    span: Span,
    key: C::Key,
    query: &QueryVtable<CTX, C::Key, C::Value>,
) -> C::Stored
where
    CTX: QueryContext,
    C: QueryCache,
    C::Key: crate::dep_graph::DepNodeParams<CTX>,
{
    try_get_cached(
        tcx,
        state,
        key,
        |value, index| {
            tcx.dep_graph().read_index(index);
            value.clone()
        },
        |key, lookup| try_execute_query(tcx, state, span, key, lookup, query),
    )
}

/// Ensure that either this query has all green inputs or been executed.
/// Executing `query::ensure(D)` is considered a read of the dep-node `D`.
///
/// This function is particularly useful when executing passes for their
/// side-effects -- e.g., in order to report errors for erroneous programs.
///
/// Note: The optimization is only available during incr. comp.
#[inline(never)]
fn ensure_query_impl<CTX, C>(
    tcx: CTX,
    state: &QueryState<CTX::DepKind, CTX::Query, C>,
    key: C::Key,
    query: &QueryVtable<CTX, C::Key, C::Value>,
) where
    C: QueryCache,
    C::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    if query.eval_always {
        let _ = get_query_impl(tcx, state, DUMMY_SP, key, query);
        return;
    }

    // Ensuring an anonymous query makes no sense
    assert!(!query.anon);

    let dep_node = query.to_dep_node(tcx, &key);

    match tcx.dep_graph().try_mark_green_and_read(tcx, &dep_node) {
        None => {
            // A None return from `try_mark_green_and_read` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.
            let _ = get_query_impl(tcx, state, DUMMY_SP, key, query);
        }
        Some((_, dep_node_index)) => {
            tcx.profiler().query_cache_hit(dep_node_index.into());
        }
    }
}

#[inline(never)]
fn force_query_impl<CTX, C>(
    tcx: CTX,
    state: &QueryState<CTX::DepKind, CTX::Query, C>,
    key: C::Key,
    span: Span,
    dep_node: DepNode<CTX::DepKind>,
    query: &QueryVtable<CTX, C::Key, C::Value>,
) where
    C: QueryCache,
    C::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.

    try_get_cached(
        tcx,
        state,
        key,
        |_, _| {
            // Cache hit, do nothing
        },
        |key, lookup| {
            let job = match JobOwner::<'_, CTX::DepKind, CTX::Query, C>::try_start(
                tcx, state, span, &key, lookup, query,
            ) {
                TryGetJob::NotYetStarted(job) => job,
                TryGetJob::Cycle(_) => return,
                #[cfg(parallel_compiler)]
                TryGetJob::JobCompleted(_) => return,
            };
            force_query_with_job(tcx, key, job, dep_node, query);
        },
    );
}

#[inline(always)]
pub fn get_query<Q, CTX>(tcx: CTX, span: Span, key: Q::Key) -> Q::Stored
where
    Q: QueryDescription<CTX>,
    Q::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    debug!("ty::query::get_query<{}>(key={:?}, span={:?})", Q::NAME, key, span);

    get_query_impl(tcx, Q::query_state(tcx), span, key, &Q::VTABLE)
}

#[inline(always)]
pub fn ensure_query<Q, CTX>(tcx: CTX, key: Q::Key)
where
    Q: QueryDescription<CTX>,
    Q::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    ensure_query_impl(tcx, Q::query_state(tcx), key, &Q::VTABLE)
}

#[inline(always)]
pub fn force_query<Q, CTX>(tcx: CTX, key: Q::Key, span: Span, dep_node: DepNode<CTX::DepKind>)
where
    Q: QueryDescription<CTX>,
    Q::Key: crate::dep_graph::DepNodeParams<CTX>,
    CTX: QueryContext,
{
    force_query_impl(tcx, Q::query_state(tcx), key, span, dep_node, &Q::VTABLE)
}
