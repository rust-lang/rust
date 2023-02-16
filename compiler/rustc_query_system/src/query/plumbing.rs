//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepContext, DepKind, DepNode, DepNodeIndex};
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::job::{report_cycle, QueryInfo, QueryJob, QueryJobId, QueryJobInfo};
use crate::query::{QueryContext, QueryMap, QuerySideEffects, QueryStackFrame};
use crate::values::Value;
use crate::HandleCycleError;
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
use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ptr;
use thin_vec::ThinVec;

use super::QueryConfig;

pub struct QueryState<K, D: DepKind> {
    #[cfg(parallel_compiler)]
    active: Sharded<FxHashMap<K, QueryResult<D>>>,
    #[cfg(not(parallel_compiler))]
    active: Lock<FxHashMap<K, QueryResult<D>>>,
}

/// Indicates the state of a query for a given key in a query map.
enum QueryResult<D: DepKind> {
    /// An already executing query. The query job can be used to await for its completion.
    Started(QueryJob<D>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

impl<K, D> QueryState<K, D>
where
    K: Eq + Hash + Clone + Debug,
    D: DepKind,
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

    pub fn try_collect_active_jobs<Qcx: Copy>(
        &self,
        qcx: Qcx,
        make_query: fn(Qcx, K) -> QueryStackFrame<D>,
        jobs: &mut QueryMap<D>,
    ) -> Option<()> {
        #[cfg(parallel_compiler)]
        {
            // We use try_lock_shards here since we are called from the
            // deadlock handler, and this shouldn't be locked.
            let shards = self.active.try_lock_shards()?;
            for shard in shards.iter() {
                for (k, v) in shard.iter() {
                    if let QueryResult::Started(ref job) = *v {
                        let query = make_query(qcx, k.clone());
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
                    let query = make_query(qcx, k.clone());
                    jobs.insert(job.id, QueryJobInfo { query, job: job.clone() });
                }
            }
        }

        Some(())
    }
}

impl<K, D: DepKind> Default for QueryState<K, D> {
    fn default() -> QueryState<K, D> {
        QueryState { active: Default::default() }
    }
}

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
struct JobOwner<'tcx, K, D: DepKind>
where
    K: Eq + Hash + Clone,
{
    state: &'tcx QueryState<K, D>,
    key: K,
    id: QueryJobId,
}

#[cold]
#[inline(never)]
fn mk_cycle<Qcx, R, D: DepKind>(
    qcx: Qcx,
    cycle_error: CycleError<D>,
    handler: HandleCycleError,
) -> R
where
    Qcx: QueryContext + crate::query::HasDepContext<DepKind = D>,
    R: std::fmt::Debug + Value<Qcx::DepContext, Qcx::DepKind>,
{
    let error = report_cycle(qcx.dep_context().sess(), &cycle_error);
    handle_cycle_error(*qcx.dep_context(), &cycle_error, error, handler)
}

fn handle_cycle_error<Tcx, V>(
    tcx: Tcx,
    cycle_error: &CycleError<Tcx::DepKind>,
    mut error: DiagnosticBuilder<'_, ErrorGuaranteed>,
    handler: HandleCycleError,
) -> V
where
    Tcx: DepContext,
    V: Value<Tcx, Tcx::DepKind>,
{
    use HandleCycleError::*;
    match handler {
        Error => {
            error.emit();
            Value::from_cycle_error(tcx, &cycle_error.cycle)
        }
        Fatal => {
            error.emit();
            tcx.sess().abort_if_errors();
            unreachable!()
        }
        DelayBug => {
            error.delay_as_bug();
            Value::from_cycle_error(tcx, &cycle_error.cycle)
        }
    }
}

impl<'tcx, K, D: DepKind> JobOwner<'tcx, K, D>
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
    fn try_start<'b, Qcx>(
        qcx: &'b Qcx,
        state: &'b QueryState<K, Qcx::DepKind>,
        span: Span,
        key: K,
    ) -> TryGetJob<'b, K, D>
    where
        Qcx: QueryContext + crate::query::HasDepContext<DepKind = D>,
    {
        #[cfg(parallel_compiler)]
        let mut state_lock = state.active.get_shard_by_value(&key).lock();
        #[cfg(not(parallel_compiler))]
        let mut state_lock = state.active.lock();
        let lock = &mut *state_lock;

        match lock.entry(key) {
            Entry::Vacant(entry) => {
                let id = qcx.next_job_id();
                let job = qcx.current_query_job();
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
                            qcx.try_collect_active_jobs().unwrap(),
                            &qcx.current_query_job(),
                            span,
                        ));
                    }
                    #[cfg(parallel_compiler)]
                    QueryResult::Started(job) => {
                        // For parallel queries, we'll block and wait until the query running
                        // in another thread has completed. Record how long we wait in the
                        // self-profiler.
                        let query_blocked_prof_timer = qcx.dep_context().profiler().query_blocked();

                        // Get the latch out
                        let latch = job.latch();

                        drop(state_lock);

                        // With parallel queries we might just have to wait on some other
                        // thread.
                        let result = latch.wait_on(qcx.current_query_job(), span);

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
    fn complete<C>(self, cache: &C, result: C::Value, dep_node_index: DepNodeIndex)
    where
        C: QueryCache<Key = K>,
    {
        // We can move out of `self` here because we `mem::forget` it below
        let key = unsafe { ptr::read(&self.key) };
        let state = self.state;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

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
        cache.complete(key, result, dep_node_index);

        job.signal_complete();
    }
}

impl<'tcx, K, D> Drop for JobOwner<'tcx, K, D>
where
    K: Eq + Hash + Clone,
    D: DepKind,
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
pub(crate) struct CycleError<D: DepKind> {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, QueryStackFrame<D>)>,
    pub cycle: Vec<QueryInfo<D>>,
}

/// The result of `try_start`.
enum TryGetJob<'tcx, K, D>
where
    K: Eq + Hash + Clone,
    D: DepKind,
{
    /// The query is not yet started. Contains a guard to the cache eventually used to start it.
    NotYetStarted(JobOwner<'tcx, K, D>),

    /// The query was already completed.
    /// Returns the result of the query and its dep-node index
    /// if it succeeded or a cycle error if it failed.
    #[cfg(parallel_compiler)]
    JobCompleted(TimingGuard<'tcx>),

    /// Trying to execute the query resulted in a cycle.
    Cycle(CycleError<D>),
}

/// Checks if the query is already computed and in the cache.
/// It returns the shard index and a lock guard to the shard,
/// which will be used if the query is not in the cache and we need
/// to compute it.
#[inline]
pub fn try_get_cached<Tcx, C>(tcx: Tcx, cache: &C, key: &C::Key) -> Option<C::Value>
where
    C: QueryCache,
    Tcx: DepContext,
{
    match cache.lookup(&key) {
        Some((value, index)) => {
            tcx.profiler().query_cache_hit(index.into());
            tcx.dep_graph().read_index(index);
            Some(value)
        }
        None => None,
    }
}

fn try_execute_query<Q, Qcx>(
    qcx: Qcx,
    state: &QueryState<Q::Key, Qcx::DepKind>,
    cache: &Q::Cache,
    span: Span,
    key: Q::Key,
    dep_node: Option<DepNode<Qcx::DepKind>>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    match JobOwner::<'_, Q::Key, Qcx::DepKind>::try_start(&qcx, state, span, key.clone()) {
        TryGetJob::NotYetStarted(job) => {
            let (result, dep_node_index) =
                execute_job::<Q, Qcx>(qcx, key.clone(), dep_node, job.id);
            if Q::FEEDABLE {
                // We may have put a value inside the cache from inside the execution.
                // Verify that it has the same hash as what we have now, to ensure consistency.
                if let Some((cached_result, _)) = cache.lookup(&key) {
                    let hasher = Q::HASH_RESULT.expect("feedable forbids no_hash");

                    let old_hash = qcx.dep_context().with_stable_hashing_context(|mut hcx| {
                        hasher(&mut hcx, cached_result.borrow())
                    });
                    let new_hash = qcx
                        .dep_context()
                        .with_stable_hashing_context(|mut hcx| hasher(&mut hcx, &result));
                    debug_assert_eq!(
                        old_hash,
                        new_hash,
                        "Computed query value for {:?}({:?}) is inconsistent with fed value,\ncomputed={:#?}\nfed={:#?}",
                        Q::DEP_KIND,
                        key,
                        result,
                        cached_result,
                    );
                }
            }
            job.complete(cache, result, dep_node_index);
            (result, Some(dep_node_index))
        }
        TryGetJob::Cycle(error) => {
            let result = mk_cycle(qcx, error, Q::HANDLE_CYCLE_ERROR);
            (result, None)
        }
        #[cfg(parallel_compiler)]
        TryGetJob::JobCompleted(query_blocked_prof_timer) => {
            let Some((v, index)) = cache.lookup(&key) else {
                panic!("value must be in cache after waiting")
            };

            qcx.dep_context().profiler().query_cache_hit(index.into());
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
    }
}

fn execute_job<Q, Qcx>(
    qcx: Qcx,
    key: Q::Key,
    mut dep_node_opt: Option<DepNode<Qcx::DepKind>>,
    job_id: QueryJobId,
) -> (Q::Value, DepNodeIndex)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    let dep_graph = qcx.dep_context().dep_graph();

    // Fast path for when incr. comp. is off.
    if !dep_graph.is_fully_enabled() {
        let prof_timer = qcx.dep_context().profiler().query_provider();
        let result = qcx.start_query(job_id, Q::DEPTH_LIMIT, None, || Q::compute(qcx, key));
        let dep_node_index = dep_graph.next_virtual_depnode_index();
        prof_timer.finish_with_query_invocation_id(dep_node_index.into());
        return (result, dep_node_index);
    }

    if !Q::ANON && !Q::EVAL_ALWAYS {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node =
            dep_node_opt.get_or_insert_with(|| Q::construct_dep_node(*qcx.dep_context(), &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = qcx.start_query(job_id, false, None, || {
            try_load_from_disk_and_cache_in_memory::<Q, Qcx>(qcx, &key, &dep_node)
        }) {
            return ret;
        }
    }

    let prof_timer = qcx.dep_context().profiler().query_provider();
    let diagnostics = Lock::new(ThinVec::new());

    let (result, dep_node_index) =
        qcx.start_query(job_id, Q::DEPTH_LIMIT, Some(&diagnostics), || {
            if Q::ANON {
                return dep_graph
                    .with_anon_task(*qcx.dep_context(), Q::DEP_KIND, || Q::compute(qcx, key));
            }

            // `to_dep_node` is expensive for some `DepKind`s.
            let dep_node =
                dep_node_opt.unwrap_or_else(|| Q::construct_dep_node(*qcx.dep_context(), &key));

            dep_graph.with_task(dep_node, qcx, key, Q::compute, Q::HASH_RESULT)
        });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    let diagnostics = diagnostics.into_inner();
    let side_effects = QuerySideEffects { diagnostics };

    if std::intrinsics::unlikely(!side_effects.is_empty()) {
        if Q::ANON {
            qcx.store_side_effects_for_anon_node(dep_node_index, side_effects);
        } else {
            qcx.store_side_effects(dep_node_index, side_effects);
        }
    }

    (result, dep_node_index)
}

fn try_load_from_disk_and_cache_in_memory<Q, Qcx>(
    qcx: Qcx,
    key: &Q::Key,
    dep_node: &DepNode<Qcx::DepKind>,
) -> Option<(Q::Value, DepNodeIndex)>
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    let dep_graph = qcx.dep_context().dep_graph();
    let (prev_dep_node_index, dep_node_index) = dep_graph.try_mark_green(qcx, &dep_node)?;

    debug_assert!(dep_graph.is_green(dep_node));

    // First we try to load the result from the on-disk cache.
    // Some things are never cached on disk.
    if let Some(try_load_from_disk) = Q::try_load_from_disk(qcx, &key) {
        let prof_timer = qcx.dep_context().profiler().incr_cache_loading();

        // The call to `with_query_deserialization` enforces that no new `DepNodes`
        // are created during deserialization. See the docs of that method for more
        // details.
        let result =
            dep_graph.with_query_deserialization(|| try_load_from_disk(qcx, prev_dep_node_index));

        prof_timer.finish_with_query_invocation_id(dep_node_index.into());

        if let Some(result) = result {
            if std::intrinsics::unlikely(
                qcx.dep_context().sess().opts.unstable_opts.query_dep_graph,
            ) {
                dep_graph.mark_debug_loaded_from_disk(*dep_node)
            }

            let prev_fingerprint = qcx
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
                try_verify || qcx.dep_context().sess().opts.unstable_opts.incremental_verify_ich,
            ) {
                incremental_verify_ich(*qcx.dep_context(), &result, dep_node, Q::HASH_RESULT);
            }

            return Some((result, dep_node_index));
        }

        // We always expect to find a cached result for things that
        // can be forced from `DepNode`.
        debug_assert!(
            !qcx.dep_context().fingerprint_style(dep_node.kind).reconstructible(),
            "missing on-disk cache entry for {dep_node:?}"
        );
    }

    // We could not load a result from the on-disk cache, so
    // recompute.
    let prof_timer = qcx.dep_context().profiler().query_provider();

    // The dep-graph for this computation is already in-place.
    let result = dep_graph.with_ignore(|| Q::compute(qcx, key.clone()));

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
    incremental_verify_ich(*qcx.dep_context(), &result, dep_node, Q::HASH_RESULT);

    Some((result, dep_node_index))
}

#[instrument(skip(tcx, result, hash_result), level = "debug")]
pub(crate) fn incremental_verify_ich<Tcx, V: Debug>(
    tcx: Tcx,
    result: &V,
    dep_node: &DepNode<Tcx::DepKind>,
    hash_result: Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>,
) -> Fingerprint
where
    Tcx: DepContext,
{
    assert!(
        tcx.dep_graph().is_green(dep_node),
        "fingerprint for green query instance not loaded from cache: {dep_node:?}",
    );

    let new_hash = hash_result.map_or(Fingerprint::ZERO, |f| {
        tcx.with_stable_hashing_context(|mut hcx| f(&mut hcx, result))
    });

    let old_hash = tcx.dep_graph().prev_fingerprint_of(dep_node);

    if Some(new_hash) != old_hash {
        incremental_verify_ich_failed(
            tcx.sess(),
            DebugArg::from(&dep_node),
            DebugArg::from(&result),
        );
    }

    new_hash
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
fn incremental_verify_ich_failed(sess: &Session, dep_node: DebugArg<'_>, result: DebugArg<'_>) {
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
        let run_cmd = if let Some(crate_name) = &sess.opts.crate_name {
            format!("`cargo clean -p {crate_name}` or `cargo clean`")
        } else {
            "`cargo clean`".to_string()
        };

        sess.emit_err(crate::error::IncrementCompilation {
            run_cmd,
            dep_node: format!("{dep_node:?}"),
        });
        panic!("Found unstable fingerprints for {dep_node:?}: {result:?}");
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
fn ensure_must_run<Q, Qcx>(qcx: Qcx, key: &Q::Key) -> (bool, Option<DepNode<Qcx::DepKind>>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    if Q::EVAL_ALWAYS {
        return (true, None);
    }

    // Ensuring an anonymous query makes no sense
    assert!(!Q::ANON);

    let dep_node = Q::construct_dep_node(*qcx.dep_context(), key);

    let dep_graph = qcx.dep_context().dep_graph();
    match dep_graph.try_mark_green(qcx, &dep_node) {
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
            qcx.dep_context().profiler().query_cache_hit(dep_node_index.into());
            (false, None)
        }
    }
}

#[derive(Debug)]
pub enum QueryMode {
    Get,
    Ensure,
}

pub fn get_query<Q, Qcx, D>(qcx: Qcx, span: Span, key: Q::Key, mode: QueryMode) -> Option<Q::Value>
where
    D: DepKind,
    Q: QueryConfig<Qcx>,
    Q::Value: Value<Qcx::DepContext, D>,
    Qcx: QueryContext,
{
    let dep_node = if let QueryMode::Ensure = mode {
        let (must_run, dep_node) = ensure_must_run::<Q, _>(qcx, &key);
        if !must_run {
            return None;
        }
        dep_node
    } else {
        None
    };

    let (result, dep_node_index) = try_execute_query::<Q, Qcx>(
        qcx,
        Q::query_state(qcx),
        Q::query_cache(qcx),
        span,
        key,
        dep_node,
    );
    if let Some(dep_node_index) = dep_node_index {
        qcx.dep_context().dep_graph().read_index(dep_node_index)
    }
    Some(result)
}

pub fn force_query<Q, Qcx, D>(qcx: Qcx, key: Q::Key, dep_node: DepNode<Qcx::DepKind>)
where
    D: DepKind,
    Q: QueryConfig<Qcx>,
    Q::Value: Value<Qcx::DepContext, D>,
    Qcx: QueryContext,
{
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    let cache = Q::query_cache(qcx);
    if let Some((_, index)) = cache.lookup(&key) {
        qcx.dep_context().profiler().query_cache_hit(index.into());
        return;
    }

    let state = Q::query_state(qcx);
    debug_assert!(!Q::ANON);

    try_execute_query::<Q, _>(qcx, state, cache, DUMMY_SP, key, Some(dep_node));
}
