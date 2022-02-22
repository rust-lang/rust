//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepContext, DepNodeIndex};
use crate::query::caches::QueryCache;
use crate::query::config::QueryDescription;
use crate::query::job::{report_cycle, QueryInfo, QueryJob, QueryJobId, QueryJobInfo};
use crate::query::{QueryContext, QueryMap, QueryStackFrame};
use rustc_data_structures::fx::{FxHashMap, FxHasher};
#[cfg(parallel_compiler)]
use rustc_data_structures::profiling::TimingGuard;
use rustc_data_structures::sharded::{get_shard_index_by_hash, Sharded};
use rustc_data_structures::sync::LockGuard;
use rustc_errors::{DiagnosticBuilder, FatalError};
use rustc_span::Span;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;

pub struct QueryCacheStore<C: QueryCache> {
    cache: C,
    shards: Sharded<C::Sharded>,
}

impl<C: QueryCache + Default> Default for QueryCacheStore<C> {
    fn default() -> Self {
        Self { cache: C::default(), shards: Default::default() }
    }
}

/// Values used when checking a query cache which can be reused on a cache-miss to execute the query.
pub struct QueryLookup {
    pub(super) key_hash: u64,
    shard: usize,
}

// We compute the key's hash once and then use it for both the
// shard lookup and the hashmap lookup. This relies on the fact
// that both of them use `FxHasher`.
fn hash_for_shard<K: Hash>(key: &K) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

impl<C: QueryCache> QueryCacheStore<C> {
    pub(super) fn get_lookup<'tcx>(
        &'tcx self,
        key: &C::Key,
    ) -> (QueryLookup, LockGuard<'tcx, C::Sharded>) {
        let key_hash = hash_for_shard(key);
        let shard = get_shard_index_by_hash(key_hash);
        let lock = self.shards.get_shard_by_index(shard).lock();
        (QueryLookup { key_hash, shard }, lock)
    }

    pub fn iter_results(&self, f: &mut dyn FnMut(&C::Key, &C::Value, DepNodeIndex)) {
        self.cache.iter(&self.shards, f)
    }
}

struct QueryStateShard<K> {
    active: FxHashMap<K, QueryResult>,
}

impl<K> Default for QueryStateShard<K> {
    fn default() -> QueryStateShard<K> {
        QueryStateShard { active: Default::default() }
    }
}

pub struct QueryState<K> {
    shards: Sharded<QueryStateShard<K>>,
}

/// Indicates the state of a query for a given key in a query map.
enum QueryResult {
    /// An already executing query. The query job can be used to await for its completion.
    Started(QueryJob),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

impl<K> QueryState<K>
where
    K: Eq + Hash + Clone + Debug,
{
    pub fn all_inactive(&self) -> bool {
        let shards = self.shards.lock_shards();
        shards.iter().all(|shard| shard.active.is_empty())
    }

    pub fn try_collect_active_jobs<CTX: Copy>(
        &self,
        tcx: CTX,
        make_query: fn(CTX, K) -> QueryStackFrame,
        jobs: &mut QueryMap,
    ) -> Option<()> {
        // We use try_lock_shards here since we are called from the
        // deadlock handler, and this shouldn't be locked.
        let shards = self.shards.try_lock_shards()?;
        for shard in shards.iter() {
            for (k, v) in shard.active.iter() {
                if let QueryResult::Started(ref job) = *v {
                    let query = make_query(tcx, k.clone());
                    jobs.insert(job.id, QueryJobInfo { query, job: job.clone() });
                }
            }
        }

        Some(())
    }
}

impl<K> Default for QueryState<K> {
    fn default() -> QueryState<K> {
        QueryState { shards: Default::default() }
    }
}

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
struct JobOwner<'tcx, K>
where
    K: Eq + Hash + Clone,
{
    state: &'tcx QueryState<K>,
    key: K,
    id: QueryJobId,
}

#[cold]
#[inline(never)]
fn mk_cycle<CTX, V, R>(
    tcx: CTX,
    error: CycleError,
    handle_cycle_error: fn(CTX, DiagnosticBuilder<'_>) -> V,
    cache: &dyn crate::query::QueryStorage<Value = V, Stored = R>,
) -> R
where
    CTX: QueryContext,
    V: std::fmt::Debug,
    R: Clone,
{
    let error = report_cycle(tcx.dep_context().sess(), error);
    let value = handle_cycle_error(tcx, error);
    cache.store_nocache(value)
}

impl<'tcx, K> JobOwner<'tcx, K>
where
    K: Eq + Hash + Clone,
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
    fn try_start<'b, CTX>(
        tcx: &'b CTX,
        state: &'b QueryState<K>,
        span: Span,
        key: K,
        lookup: QueryLookup,
    ) -> TryGetJob<'b, K>
    where
        CTX: QueryContext,
    {
        let shard = lookup.shard;
        let mut state_lock = state.shards.get_shard_by_index(shard).lock();
        let lock = &mut *state_lock;

        match lock.active.entry(key) {
            Entry::Vacant(entry) => {
                let id = tcx.next_job_id();
                let job = tcx.current_query_job();
                let job = QueryJob::new(id, span, job);

                let key = entry.key().clone();
                entry.insert(QueryResult::Started(job));

                let owner = JobOwner { state, id, key };
                return TryGetJob::NotYetStarted(owner);
            }
            Entry::Occupied(mut entry) => {
                match entry.get_mut() {
                    #[cfg(not(parallel_compiler))]
                    QueryResult::Started(job) => {
                        let id = job.id;
                        drop(state_lock);

                        // If we are single-threaded we know that we have cycle error,
                        // so we just return the error.
                        return TryGetJob::Cycle(id.find_cycle_in_stack(
                            tcx.try_collect_active_jobs().unwrap(),
                            &tcx.current_query_job(),
                            span,
                        ));
                    }
                    #[cfg(parallel_compiler)]
                    QueryResult::Started(job) => {
                        // For parallel queries, we'll block and wait until the query running
                        // in another thread has completed. Record how long we wait in the
                        // self-profiler.
                        let query_blocked_prof_timer = tcx.dep_context().profiler().query_blocked();

                        // Get the latch out
                        let latch = job.latch();

                        drop(state_lock);

                        // With parallel queries we might just have to wait on some other
                        // thread.
                        let result = latch.wait_on(tcx.current_query_job(), span);

                        match result {
                            Ok(()) => TryGetJob::JobCompleted(query_blocked_prof_timer),
                            Err(cycle) => TryGetJob::Cycle(cycle),
                        }
                    }
                    QueryResult::Poisoned => FatalError.raise(),
                }
            }
        }
    }

    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter and forgets the JobOwner, so it won't poison the query
    fn complete<C>(
        self,
        cache: &QueryCacheStore<C>,
        result: C::Value,
        dep_node_index: DepNodeIndex,
    ) -> C::Stored
    where
        C: QueryCache<Key = K>,
    {
        // We can move out of `self` here because we `mem::forget` it below
        let key = unsafe { ptr::read(&self.key) };
        let state = self.state;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

        let (job, result) = {
            let key_hash = hash_for_shard(&key);
            let shard = get_shard_index_by_hash(key_hash);
            let job = {
                let mut lock = state.shards.get_shard_by_index(shard).lock();
                match lock.active.remove(&key).unwrap() {
                    QueryResult::Started(job) => job,
                    QueryResult::Poisoned => panic!(),
                }
            };
            let result = {
                let mut lock = cache.shards.get_shard_by_index(shard).lock();
                cache.cache.complete(&mut lock, key, result, dep_node_index)
            };
            (job, result)
        };

        job.signal_complete();
        result
    }
}

impl<'tcx, K> Drop for JobOwner<'tcx, K>
where
    K: Eq + Hash + Clone,
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
pub(crate) struct CycleError {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, QueryStackFrame)>,
    pub cycle: Vec<QueryInfo>,
}

/// The result of `try_start`.
enum TryGetJob<'tcx, K>
where
    K: Eq + Hash + Clone,
{
    /// The query is not yet started. Contains a guard to the cache eventually used to start it.
    NotYetStarted(JobOwner<'tcx, K>),

    /// The query was already completed.
    /// Returns the result of the query and its dep-node index
    /// if it succeeded or a cycle error if it failed.
    #[cfg(parallel_compiler)]
    JobCompleted(TimingGuard<'tcx>),

    /// Trying to execute the query resulted in a cycle.
    Cycle(CycleError),
}

/// Checks if the query is already computed and in the cache.
/// It returns the shard index and a lock guard to the shard,
/// which will be used if the query is not in the cache and we need
/// to compute it.
#[inline]
pub fn try_get_cached<'a, CTX, C, R, OnHit>(
    tcx: CTX,
    cache: &'a QueryCacheStore<C>,
    key: &C::Key,
    // `on_hit` can be called while holding a lock to the query cache
    on_hit: OnHit,
) -> Result<R, QueryLookup>
where
    C: QueryCache,
    CTX: DepContext,
    OnHit: FnOnce(&C::Stored) -> R,
{
    cache.cache.lookup(cache, &key, |value, index| {
        if unlikely!(tcx.profiler().enabled()) {
            tcx.profiler().query_cache_hit(index.into());
        }
        on_hit(value)
    })
}

pub fn get_query<Q, CTX>(tcx: CTX, span: Span, key: Q::Key, lookup: QueryLookup) -> Q::Stored
where
    Q: QueryDescription<CTX>,
    Q::Value: Debug,
    CTX: QueryContext,
{
    debug!("ty::query::get_query<{}>(key={:?}, span={:?})", Q::NAME, key, span);
    let cache = Q::query_cache(tcx);
    let compute = Q::compute(tcx, &key);
    match JobOwner::<'_, Q::Key>::try_start(&tcx, Q::query_state(tcx), span, key.clone(), lookup) {
        TryGetJob::NotYetStarted(job) => {
            let prof_timer = tcx.dep_context().profiler().query_provider();
            let result = tcx.start_query(job.id, || compute(*tcx.dep_context(), key));
            let dep_node_index = tcx.dep_context().dep_graph().next_virtual_depnode_index();
            prof_timer.finish_with_query_invocation_id(dep_node_index.into());
            job.complete(cache, result, dep_node_index)
        }
        TryGetJob::Cycle(error) => mk_cycle(tcx, error, Q::handle_cycle_error, &cache.cache),
        #[cfg(parallel_compiler)]
        TryGetJob::JobCompleted(query_blocked_prof_timer) => {
            let (v, index) = cache
                .cache
                .lookup(cache, &key, |value, index| (value.clone(), index))
                .unwrap_or_else(|_| panic!("value must be in cache after waiting"));

            if unlikely!(tcx.dep_context().profiler().enabled()) {
                tcx.dep_context().profiler().query_cache_hit(index.into());
            }
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            v
        }
    }
}
