//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepContext, DepNode, DepNodeIndex, DepNodeParams};
use crate::query::caches::QueryCache;
use crate::query::config::{QueryDescription, QueryVTable};
use crate::query::job::{report_cycle, QueryInfo, QueryJob, QueryJobId, QueryJobInfo};
use crate::query::{QueryContext, QueryMap, QuerySideEffects, QueryStackFrame};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
#[cfg(parallel_compiler)]
use rustc_data_structures::profiling::TimingGuard;
#[cfg(parallel_compiler)]
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::Lock;
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed, FatalError};
use rustc_session::Session;
use rustc_span::{Span, DUMMY_SP};
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ptr;
use thin_vec::ThinVec;

pub struct QueryState<K> {
    #[cfg(parallel_compiler)]
    active: Sharded<FxHashMap<K, QueryResult>>,
    #[cfg(not(parallel_compiler))]
    active: Lock<FxHashMap<K, QueryResult>>,
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
        #[cfg(parallel_compiler)]
        {
            let shards = self.active.lock_shards();
            shards.iter().all(|shard| shard.is_empty())
        }
        #[cfg(not(parallel_compiler))]
        {
            self.active.lock().is_empty()
        }
    }

    pub fn try_collect_active_jobs<CTX: Copy>(
        &self,
        tcx: CTX,
        make_query: fn(CTX, K) -> QueryStackFrame,
        jobs: &mut QueryMap,
    ) -> Option<()> {
        #[cfg(parallel_compiler)]
        {
            // We use try_lock_shards here since we are called from the
            // deadlock handler, and this shouldn't be locked.
            let shards = self.active.try_lock_shards()?;
            for shard in shards.iter() {
                for (k, v) in shard.iter() {
                    if let QueryResult::Started(ref job) = *v {
                        let query = make_query(tcx, k.clone());
                        jobs.insert(job.id, QueryJobInfo { query, job: job.clone() });
                    }
                }
            }
        }
        #[cfg(not(parallel_compiler))]
        {
            // We use try_lock here since we are called from the
            // deadlock handler, and this shouldn't be locked.
            // (FIXME: Is this relevant for non-parallel compilers? It doesn't
            // really hurt much.)
            for (k, v) in self.active.try_lock()?.iter() {
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
        QueryState { active: Default::default() }
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
    handle_cycle_error: fn(CTX, DiagnosticBuilder<'_, ErrorGuaranteed>) -> V,
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
    ) -> TryGetJob<'b, K>
    where
        CTX: QueryContext,
    {
        #[cfg(parallel_compiler)]
        let mut state_lock = state.active.get_shard_by_value(&key).lock();
        #[cfg(not(parallel_compiler))]
        let mut state_lock = state.active.lock();
        let lock = &mut *state_lock;

        match lock.entry(key) {
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
    fn complete<C>(self, cache: &C, result: C::Value, dep_node_index: DepNodeIndex) -> C::Stored
    where
        C: QueryCache<Key = K>,
    {
        // We can move out of `self` here because we `mem::forget` it below
        let key = unsafe { ptr::read(&self.key) };
        let state = self.state;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

        let (job, result) = {
            let job = {
                #[cfg(parallel_compiler)]
                let mut lock = state.active.get_shard_by_value(&key).lock();
                #[cfg(not(parallel_compiler))]
                let mut lock = state.active.lock();
                match lock.remove(&key).unwrap() {
                    QueryResult::Started(job) => job,
                    QueryResult::Poisoned => panic!(),
                }
            };
            let result = cache.complete(key, result, dep_node_index);
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
        let job = {
            #[cfg(parallel_compiler)]
            let mut shard = state.active.get_shard_by_value(&self.key).lock();
            #[cfg(not(parallel_compiler))]
            let mut shard = state.active.lock();
            let job = match shard.remove(&self.key).unwrap() {
                QueryResult::Started(job) => job,
                QueryResult::Poisoned => panic!(),
            };
            shard.insert(self.key.clone(), QueryResult::Poisoned);
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
    cache: &'a C,
    key: &C::Key,
    // `on_hit` can be called while holding a lock to the query cache
    on_hit: OnHit,
) -> Result<R, ()>
where
    C: QueryCache,
    CTX: DepContext,
    OnHit: FnOnce(&C::Stored) -> R,
{
    cache.lookup(&key, |value, index| {
        if std::intrinsics::unlikely(tcx.profiler().enabled()) {
            tcx.profiler().query_cache_hit(index.into());
        }
        tcx.dep_graph().read_index(index);
        on_hit(value)
    })
}

fn try_execute_query<CTX, C>(
    tcx: CTX,
    state: &QueryState<C::Key>,
    cache: &C,
    span: Span,
    key: C::Key,
    dep_node: Option<DepNode<CTX::DepKind>>,
    query: &QueryVTable<CTX, C::Key, C::Value>,
) -> (C::Stored, Option<DepNodeIndex>)
where
    C: QueryCache,
    C::Key: Clone + DepNodeParams<CTX::DepContext>,
    CTX: QueryContext,
{
    match JobOwner::<'_, C::Key>::try_start(&tcx, state, span, key.clone()) {
        TryGetJob::NotYetStarted(job) => {
            let (result, dep_node_index) = execute_job(tcx, key, dep_node, query, job.id);
            let result = job.complete(cache, result, dep_node_index);
            (result, Some(dep_node_index))
        }
        TryGetJob::Cycle(error) => {
            let result = mk_cycle(tcx, error, query.handle_cycle_error, cache);
            (result, None)
        }
        #[cfg(parallel_compiler)]
        TryGetJob::JobCompleted(query_blocked_prof_timer) => {
            let (v, index) = cache
                .lookup(&key, |value, index| (value.clone(), index))
                .unwrap_or_else(|_| panic!("value must be in cache after waiting"));

            if std::intrinsics::unlikely(tcx.dep_context().profiler().enabled()) {
                tcx.dep_context().profiler().query_cache_hit(index.into());
            }
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
    }
}

fn execute_job<CTX, K, V>(
    tcx: CTX,
    key: K,
    mut dep_node_opt: Option<DepNode<CTX::DepKind>>,
    query: &QueryVTable<CTX, K, V>,
    job_id: QueryJobId,
) -> (V, DepNodeIndex)
where
    K: Clone + DepNodeParams<CTX::DepContext>,
    V: Debug,
    CTX: QueryContext,
{
    let dep_graph = tcx.dep_context().dep_graph();

    // Fast path for when incr. comp. is off.
    if !dep_graph.is_fully_enabled() {
        let prof_timer = tcx.dep_context().profiler().query_provider();
        let result = tcx.start_query(job_id, query.depth_limit, None, || {
            query.compute(*tcx.dep_context(), key)
        });
        let dep_node_index = dep_graph.next_virtual_depnode_index();
        prof_timer.finish_with_query_invocation_id(dep_node_index.into());
        return (result, dep_node_index);
    }

    if !query.anon && !query.eval_always {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node =
            dep_node_opt.get_or_insert_with(|| query.to_dep_node(*tcx.dep_context(), &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = tcx.start_query(job_id, false, None, || {
            try_load_from_disk_and_cache_in_memory(tcx, &key, &dep_node, query)
        }) {
            return ret;
        }
    }

    let prof_timer = tcx.dep_context().profiler().query_provider();
    let diagnostics = Lock::new(ThinVec::new());

    let (result, dep_node_index) =
        tcx.start_query(job_id, query.depth_limit, Some(&diagnostics), || {
            if query.anon {
                return dep_graph.with_anon_task(*tcx.dep_context(), query.dep_kind, || {
                    query.compute(*tcx.dep_context(), key)
                });
            }

            // `to_dep_node` is expensive for some `DepKind`s.
            let dep_node =
                dep_node_opt.unwrap_or_else(|| query.to_dep_node(*tcx.dep_context(), &key));

            dep_graph.with_task(dep_node, *tcx.dep_context(), key, query.compute, query.hash_result)
        });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    let diagnostics = diagnostics.into_inner();
    let side_effects = QuerySideEffects { diagnostics };

    if std::intrinsics::unlikely(!side_effects.is_empty()) {
        if query.anon {
            tcx.store_side_effects_for_anon_node(dep_node_index, side_effects);
        } else {
            tcx.store_side_effects(dep_node_index, side_effects);
        }
    }

    (result, dep_node_index)
}

fn try_load_from_disk_and_cache_in_memory<CTX, K, V>(
    tcx: CTX,
    key: &K,
    dep_node: &DepNode<CTX::DepKind>,
    query: &QueryVTable<CTX, K, V>,
) -> Option<(V, DepNodeIndex)>
where
    K: Clone,
    CTX: QueryContext,
    V: Debug,
{
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    let dep_graph = tcx.dep_context().dep_graph();
    let (prev_dep_node_index, dep_node_index) = dep_graph.try_mark_green(tcx, &dep_node)?;

    debug_assert!(dep_graph.is_green(dep_node));

    // First we try to load the result from the on-disk cache.
    // Some things are never cached on disk.
    if query.cache_on_disk {
        let prof_timer = tcx.dep_context().profiler().incr_cache_loading();

        // The call to `with_query_deserialization` enforces that no new `DepNodes`
        // are created during deserialization. See the docs of that method for more
        // details.
        let result = dep_graph
            .with_query_deserialization(|| query.try_load_from_disk(tcx, prev_dep_node_index));

        prof_timer.finish_with_query_invocation_id(dep_node_index.into());

        if let Some(result) = result {
            if std::intrinsics::unlikely(
                tcx.dep_context().sess().opts.unstable_opts.query_dep_graph,
            ) {
                dep_graph.mark_debug_loaded_from_disk(*dep_node)
            }

            let prev_fingerprint = tcx
                .dep_context()
                .dep_graph()
                .prev_fingerprint_of(dep_node)
                .unwrap_or(Fingerprint::ZERO);
            // If `-Zincremental-verify-ich` is specified, re-hash results from
            // the cache and make sure that they have the expected fingerprint.
            //
            // If not, we still seek to verify a subset of fingerprints loaded
            // from disk. Re-hashing results is fairly expensive, so we can't
            // currently afford to verify every hash. This subset should still
            // give us some coverage of potential bugs though.
            let try_verify = prev_fingerprint.as_value().1 % 32 == 0;
            if std::intrinsics::unlikely(
                try_verify || tcx.dep_context().sess().opts.unstable_opts.incremental_verify_ich,
            ) {
                incremental_verify_ich(*tcx.dep_context(), &result, dep_node, query);
            }

            return Some((result, dep_node_index));
        }

        // We always expect to find a cached result for things that
        // can be forced from `DepNode`.
        debug_assert!(
            !tcx.dep_context().fingerprint_style(dep_node.kind).reconstructible(),
            "missing on-disk cache entry for {:?}",
            dep_node
        );
    }

    // We could not load a result from the on-disk cache, so
    // recompute.
    let prof_timer = tcx.dep_context().profiler().query_provider();

    // The dep-graph for this computation is already in-place.
    let result = dep_graph.with_ignore(|| query.compute(*tcx.dep_context(), key.clone()));

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Verify that re-running the query produced a result with the expected hash
    // This catches bugs in query implementations, turning them into ICEs.
    // For example, a query might sort its result by `DefId` - since `DefId`s are
    // not stable across compilation sessions, the result could get up getting sorted
    // in a different order when the query is re-run, even though all of the inputs
    // (e.g. `DefPathHash` values) were green.
    //
    // See issue #82920 for an example of a miscompilation that would get turned into
    // an ICE by this check
    incremental_verify_ich(*tcx.dep_context(), &result, dep_node, query);

    Some((result, dep_node_index))
}

fn incremental_verify_ich<CTX, K, V: Debug>(
    tcx: CTX::DepContext,
    result: &V,
    dep_node: &DepNode<CTX::DepKind>,
    query: &QueryVTable<CTX, K, V>,
) where
    CTX: QueryContext,
{
    assert!(
        tcx.dep_graph().is_green(dep_node),
        "fingerprint for green query instance not loaded from cache: {:?}",
        dep_node,
    );

    debug!("BEGIN verify_ich({:?})", dep_node);
    let new_hash = query.hash_result.map_or(Fingerprint::ZERO, |f| {
        tcx.with_stable_hashing_context(|mut hcx| f(&mut hcx, result))
    });
    let old_hash = tcx.dep_graph().prev_fingerprint_of(dep_node);
    debug!("END verify_ich({:?})", dep_node);

    if Some(new_hash) != old_hash {
        incremental_verify_ich_cold(tcx.sess(), DebugArg::from(&dep_node), DebugArg::from(&result));
    }
}

// This DebugArg business is largely a mirror of std::fmt::ArgumentV1, which is
// currently not exposed publicly.
//
// The PR which added this attempted to use `&dyn Debug` instead, but that
// showed statistically significant worse compiler performance. It's not
// actually clear what the cause there was -- the code should be cold. If this
// can be replaced with `&dyn Debug` with on perf impact, then it probably
// should be.
extern "C" {
    type Opaque;
}

struct DebugArg<'a> {
    value: &'a Opaque,
    fmt: fn(&Opaque, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
}

impl<'a, T> From<&'a T> for DebugArg<'a>
where
    T: std::fmt::Debug,
{
    fn from(value: &'a T) -> DebugArg<'a> {
        DebugArg {
            value: unsafe { std::mem::transmute(value) },
            fmt: unsafe {
                std::mem::transmute(<T as std::fmt::Debug>::fmt as fn(_, _) -> std::fmt::Result)
            },
        }
    }
}

impl std::fmt::Debug for DebugArg<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (self.fmt)(self.value, f)
    }
}

// Note that this is marked #[cold] and intentionally takes the equivalent of
// `dyn Debug` for its arguments, as we want to avoid generating a bunch of
// different implementations for LLVM to chew on (and filling up the final
// binary, too).
#[cold]
fn incremental_verify_ich_cold(sess: &Session, dep_node: DebugArg<'_>, result: DebugArg<'_>) {
    let run_cmd = if let Some(crate_name) = &sess.opts.crate_name {
        format!("`cargo clean -p {}` or `cargo clean`", crate_name)
    } else {
        "`cargo clean`".to_string()
    };

    // When we emit an error message and panic, we try to debug-print the `DepNode`
    // and query result. Unfortunately, this can cause us to run additional queries,
    // which may result in another fingerprint mismatch while we're in the middle
    // of processing this one. To avoid a double-panic (which kills the process
    // before we can print out the query static), we print out a terse
    // but 'safe' message if we detect a re-entrant call to this method.
    thread_local! {
        static INSIDE_VERIFY_PANIC: Cell<bool> = const { Cell::new(false) };
    };

    let old_in_panic = INSIDE_VERIFY_PANIC.with(|in_panic| in_panic.replace(true));

    if old_in_panic {
        sess.emit_err(crate::error::Reentrant);
    } else {
        sess.emit_err(crate::error::IncrementCompilation {
            run_cmd,
            dep_node: format!("{:?}", dep_node),
        });
        panic!("Found unstable fingerprints for {:?}: {:?}", dep_node, result);
    }

    INSIDE_VERIFY_PANIC.with(|in_panic| in_panic.set(old_in_panic));
}

/// Ensure that either this query has all green inputs or been executed.
/// Executing `query::ensure(D)` is considered a read of the dep-node `D`.
/// Returns true if the query should still run.
///
/// This function is particularly useful when executing passes for their
/// side-effects -- e.g., in order to report errors for erroneous programs.
///
/// Note: The optimization is only available during incr. comp.
#[inline(never)]
fn ensure_must_run<CTX, K, V>(
    tcx: CTX,
    key: &K,
    query: &QueryVTable<CTX, K, V>,
) -> (bool, Option<DepNode<CTX::DepKind>>)
where
    K: crate::dep_graph::DepNodeParams<CTX::DepContext>,
    CTX: QueryContext,
{
    if query.eval_always {
        return (true, None);
    }

    // Ensuring an anonymous query makes no sense
    assert!(!query.anon);

    let dep_node = query.to_dep_node(*tcx.dep_context(), key);

    let dep_graph = tcx.dep_context().dep_graph();
    match dep_graph.try_mark_green(tcx, &dep_node) {
        None => {
            // A None return from `try_mark_green` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.
            (true, Some(dep_node))
        }
        Some((_, dep_node_index)) => {
            dep_graph.read_index(dep_node_index);
            tcx.dep_context().profiler().query_cache_hit(dep_node_index.into());
            (false, None)
        }
    }
}

#[derive(Debug)]
pub enum QueryMode {
    Get,
    Ensure,
}

pub fn get_query<Q, CTX>(tcx: CTX, span: Span, key: Q::Key, mode: QueryMode) -> Option<Q::Stored>
where
    Q: QueryDescription<CTX>,
    Q::Key: DepNodeParams<CTX::DepContext>,
    CTX: QueryContext,
{
    let query = Q::make_vtable(tcx, &key);
    let dep_node = if let QueryMode::Ensure = mode {
        let (must_run, dep_node) = ensure_must_run(tcx, &key, &query);
        if !must_run {
            return None;
        }
        dep_node
    } else {
        None
    };

    let (result, dep_node_index) = try_execute_query(
        tcx,
        Q::query_state(tcx),
        Q::query_cache(tcx),
        span,
        key,
        dep_node,
        &query,
    );
    if let Some(dep_node_index) = dep_node_index {
        tcx.dep_context().dep_graph().read_index(dep_node_index)
    }
    Some(result)
}

pub fn force_query<Q, CTX>(tcx: CTX, key: Q::Key, dep_node: DepNode<CTX::DepKind>)
where
    Q: QueryDescription<CTX>,
    Q::Key: DepNodeParams<CTX::DepContext>,
    CTX: QueryContext,
{
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    let cache = Q::query_cache(tcx);
    let cached = cache.lookup(&key, |_, index| {
        if std::intrinsics::unlikely(tcx.dep_context().profiler().enabled()) {
            tcx.dep_context().profiler().query_cache_hit(index.into());
        }
    });

    match cached {
        Ok(()) => return,
        Err(()) => {}
    }

    let query = Q::make_vtable(tcx, &key);
    let state = Q::query_state(tcx);
    debug_assert!(!query.anon);

    try_execute_query(tcx, state, cache, DUMMY_SP, key, Some(dep_node), &query);
}
