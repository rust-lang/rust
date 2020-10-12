use crate::query::plumbing::CycleError;

use rustc_data_structures::fx::FxHashMap;
use rustc_span::Span;

use std::convert::TryFrom;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroU32;

#[cfg(parallel_compiler)]
use {
    super::QueryContext,
    parking_lot::{Condvar, Mutex},
    rustc_data_structures::fx::FxHashSet,
    rustc_data_structures::stable_hasher::{HashStable, StableHasher},
    rustc_data_structures::sync::Lock,
    rustc_data_structures::sync::Lrc,
    rustc_data_structures::{jobserver, OnDrop},
    rustc_rayon_core as rayon_core,
    rustc_span::DUMMY_SP,
    std::iter::FromIterator,
    std::{mem, process},
};

/// Represents a span and a query key.
#[derive(Clone, Debug)]
pub struct QueryInfo<Q> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,
    pub query: Q,
}

pub(crate) type QueryMap<D, Q> = FxHashMap<QueryJobId<D>, QueryJobInfo<D, Q>>;

/// A value uniquely identifiying an active query job within a shard in the query cache.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QueryShardJobId(pub NonZeroU32);

/// A value uniquely identifiying an active query job.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QueryJobId<D> {
    /// Which job within a shard is this
    pub job: QueryShardJobId,

    /// In which shard is this job
    pub shard: u16,

    /// What kind of query this job is.
    pub kind: D,
}

impl<D> QueryJobId<D>
where
    D: Copy + Clone + Eq + Hash,
{
    pub fn new(job: QueryShardJobId, shard: usize, kind: D) -> Self {
        QueryJobId { job, shard: u16::try_from(shard).unwrap(), kind }
    }

    fn query<Q: Clone>(self, map: &QueryMap<D, Q>) -> Q {
        map.get(&self).unwrap().info.query.clone()
    }

    #[cfg(parallel_compiler)]
    fn span<Q: Clone>(self, map: &QueryMap<D, Q>) -> Span {
        map.get(&self).unwrap().job.span
    }

    #[cfg(parallel_compiler)]
    fn parent<Q: Clone>(self, map: &QueryMap<D, Q>) -> Option<QueryJobId<D>> {
        map.get(&self).unwrap().job.parent
    }

    #[cfg(parallel_compiler)]
    fn latch<'a, Q: Clone>(self, map: &'a QueryMap<D, Q>) -> Option<&'a QueryLatch<D, Q>> {
        map.get(&self).unwrap().job.latch.as_ref()
    }
}

pub struct QueryJobInfo<D, Q> {
    pub info: QueryInfo<Q>,
    pub job: QueryJob<D, Q>,
}

/// Represents an active query job.
#[derive(Clone)]
pub struct QueryJob<D, Q> {
    pub id: QueryShardJobId,

    /// The span corresponding to the reason for which this query was required.
    pub span: Span,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<QueryJobId<D>>,

    /// The latch that is used to wait on this job.
    #[cfg(parallel_compiler)]
    latch: Option<QueryLatch<D, Q>>,

    dummy: PhantomData<QueryLatch<D, Q>>,
}

impl<D, Q> QueryJob<D, Q>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
{
    /// Creates a new query job.
    pub fn new(id: QueryShardJobId, span: Span, parent: Option<QueryJobId<D>>) -> Self {
        QueryJob {
            id,
            span,
            parent,
            #[cfg(parallel_compiler)]
            latch: None,
            dummy: PhantomData,
        }
    }

    #[cfg(parallel_compiler)]
    pub(super) fn latch(&mut self, _id: QueryJobId<D>) -> QueryLatch<D, Q> {
        if self.latch.is_none() {
            self.latch = Some(QueryLatch::new());
        }
        self.latch.as_ref().unwrap().clone()
    }

    #[cfg(not(parallel_compiler))]
    pub(super) fn latch(&mut self, id: QueryJobId<D>) -> QueryLatch<D, Q> {
        QueryLatch { id, dummy: PhantomData }
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    pub fn signal_complete(self) {
        #[cfg(parallel_compiler)]
        {
            if let Some(latch) = self.latch {
                latch.set();
            }
        }
    }
}

#[cfg(not(parallel_compiler))]
#[derive(Clone)]
pub(super) struct QueryLatch<D, Q> {
    id: QueryJobId<D>,
    dummy: PhantomData<Q>,
}

#[cfg(not(parallel_compiler))]
impl<D, Q> QueryLatch<D, Q>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
{
    pub(super) fn find_cycle_in_stack(
        &self,
        query_map: QueryMap<D, Q>,
        current_job: &Option<QueryJobId<D>>,
        span: Span,
    ) -> CycleError<Q> {
        // Find the waitee amongst `current_job` parents
        let mut cycle = Vec::new();
        let mut current_job = Option::clone(current_job);

        while let Some(job) = current_job {
            let info = query_map.get(&job).unwrap();
            cycle.push(info.info.clone());

            if job == self.id {
                cycle.reverse();

                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = info
                    .job
                    .parent
                    .as_ref()
                    .map(|parent| (info.info.span, parent.query(&query_map)));
                return CycleError { usage, cycle };
            }

            current_job = info.job.parent;
        }

        panic!("did not find a cycle")
    }
}

#[cfg(parallel_compiler)]
struct QueryWaiter<D, Q> {
    query: Option<QueryJobId<D>>,
    condvar: Condvar,
    span: Span,
    cycle: Lock<Option<CycleError<Q>>>,
}

#[cfg(parallel_compiler)]
impl<D, Q> QueryWaiter<D, Q> {
    fn notify(&self, registry: &rayon_core::Registry) {
        rayon_core::mark_unblocked(registry);
        self.condvar.notify_one();
    }
}

#[cfg(parallel_compiler)]
struct QueryLatchInfo<D, Q> {
    complete: bool,
    waiters: Vec<Lrc<QueryWaiter<D, Q>>>,
}

#[cfg(parallel_compiler)]
#[derive(Clone)]
pub(super) struct QueryLatch<D, Q> {
    info: Lrc<Mutex<QueryLatchInfo<D, Q>>>,
}

#[cfg(parallel_compiler)]
impl<D: Eq + Hash, Q: Clone> QueryLatch<D, Q> {
    fn new() -> Self {
        QueryLatch {
            info: Lrc::new(Mutex::new(QueryLatchInfo { complete: false, waiters: Vec::new() })),
        }
    }
}

#[cfg(parallel_compiler)]
impl<D, Q> QueryLatch<D, Q> {
    /// Awaits for the query job to complete.
    pub(super) fn wait_on(
        &self,
        query: Option<QueryJobId<D>>,
        span: Span,
    ) -> Result<(), CycleError<Q>> {
        let waiter =
            Lrc::new(QueryWaiter { query, span, cycle: Lock::new(None), condvar: Condvar::new() });
        self.wait_on_inner(&waiter);
        // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
        // although another thread may still have a Lrc reference so we cannot
        // use Lrc::get_mut
        let mut cycle = waiter.cycle.lock();
        match cycle.take() {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Awaits the caller on this latch by blocking the current thread.
    fn wait_on_inner(&self, waiter: &Lrc<QueryWaiter<D, Q>>) {
        let mut info = self.info.lock();
        if !info.complete {
            // We push the waiter on to the `waiters` list. It can be accessed inside
            // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
            // Both of these will remove it from the `waiters` list before resuming
            // this thread.
            info.waiters.push(waiter.clone());

            // If this detects a deadlock and the deadlock handler wants to resume this thread
            // we have to be in the `wait` call. This is ensured by the deadlock handler
            // getting the self.info lock.
            rayon_core::mark_blocked();
            jobserver::release_thread();
            waiter.condvar.wait(&mut info);
            // Release the lock before we potentially block in `acquire_thread`
            mem::drop(info);
            jobserver::acquire_thread();
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        info.complete = true;
        let registry = rayon_core::Registry::current();
        for waiter in info.waiters.drain(..) {
            waiter.notify(&registry);
        }
    }

    /// Removes a single waiter from the list of waiters.
    /// This is used to break query cycles.
    fn extract_waiter(&self, waiter: usize) -> Lrc<QueryWaiter<D, Q>> {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        // Remove the waiter from the list of waiters
        info.waiters.remove(waiter)
    }
}

/// A resumable waiter of a query. The usize is the index into waiters in the query's latch
#[cfg(parallel_compiler)]
type Waiter<D> = (QueryJobId<D>, usize);

/// Visits all the non-resumable and resumable waiters of a query.
/// Only waiters in a query are visited.
/// `visit` is called for every waiter and is passed a query waiting on `query_ref`
/// and a span indicating the reason the query waited on `query_ref`.
/// If `visit` returns Some, this function returns.
/// For visits of non-resumable waiters it returns the return value of `visit`.
/// For visits of resumable waiters it returns Some(Some(Waiter)) which has the
/// required information to resume the waiter.
/// If all `visit` calls returns None, this function also returns None.
#[cfg(parallel_compiler)]
fn visit_waiters<D, Q, F>(
    query_map: &QueryMap<D, Q>,
    query: QueryJobId<D>,
    mut visit: F,
) -> Option<Option<Waiter<D>>>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
    F: FnMut(Span, QueryJobId<D>) -> Option<Option<Waiter<D>>>,
{
    // Visit the parent query which is a non-resumable waiter since it's on the same stack
    if let Some(parent) = query.parent(query_map) {
        if let Some(cycle) = visit(query.span(query_map), parent) {
            return Some(cycle);
        }
    }

    // Visit the explicit waiters which use condvars and are resumable
    if let Some(latch) = query.latch(query_map) {
        for (i, waiter) in latch.info.lock().waiters.iter().enumerate() {
            if let Some(waiter_query) = waiter.query {
                if visit(waiter.span, waiter_query).is_some() {
                    // Return a value which indicates that this waiter can be resumed
                    return Some(Some((query, i)));
                }
            }
        }
    }

    None
}

/// Look for query cycles by doing a depth first search starting at `query`.
/// `span` is the reason for the `query` to execute. This is initially DUMMY_SP.
/// If a cycle is detected, this initial value is replaced with the span causing
/// the cycle.
#[cfg(parallel_compiler)]
fn cycle_check<D, Q>(
    query_map: &QueryMap<D, Q>,
    query: QueryJobId<D>,
    span: Span,
    stack: &mut Vec<(Span, QueryJobId<D>)>,
    visited: &mut FxHashSet<QueryJobId<D>>,
) -> Option<Option<Waiter<D>>>
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
{
    if !visited.insert(query) {
        return if let Some(p) = stack.iter().position(|q| q.1 == query) {
            // We detected a query cycle, fix up the initial span and return Some

            // Remove previous stack entries
            stack.drain(0..p);
            // Replace the span for the first query with the cycle cause
            stack[0].0 = span;
            Some(None)
        } else {
            None
        };
    }

    // Query marked as visited is added it to the stack
    stack.push((span, query));

    // Visit all the waiters
    let r = visit_waiters(query_map, query, |span, successor| {
        cycle_check(query_map, successor, span, stack, visited)
    });

    // Remove the entry in our stack if we didn't find a cycle
    if r.is_none() {
        stack.pop();
    }

    r
}

/// Finds out if there's a path to the compiler root (aka. code which isn't in a query)
/// from `query` without going through any of the queries in `visited`.
/// This is achieved with a depth first search.
#[cfg(parallel_compiler)]
fn connected_to_root<D, Q>(
    query_map: &QueryMap<D, Q>,
    query: QueryJobId<D>,
    visited: &mut FxHashSet<QueryJobId<D>>,
) -> bool
where
    D: Copy + Clone + Eq + Hash,
    Q: Clone,
{
    // We already visited this or we're deliberately ignoring it
    if !visited.insert(query) {
        return false;
    }

    // This query is connected to the root (it has no query parent), return true
    if query.parent(query_map).is_none() {
        return true;
    }

    visit_waiters(query_map, query, |_, successor| {
        connected_to_root(query_map, successor, visited).then_some(None)
    })
    .is_some()
}

// Deterministically pick an query from a list
#[cfg(parallel_compiler)]
fn pick_query<'a, CTX, T, F>(
    query_map: &QueryMap<CTX::DepKind, CTX::Query>,
    tcx: CTX,
    queries: &'a [T],
    f: F,
) -> &'a T
where
    CTX: QueryContext,
    F: Fn(&T) -> (Span, QueryJobId<CTX::DepKind>),
{
    // Deterministically pick an entry point
    // FIXME: Sort this instead
    let mut hcx = tcx.create_stable_hashing_context();
    queries
        .iter()
        .min_by_key(|v| {
            let (span, query) = f(v);
            let mut stable_hasher = StableHasher::new();
            query.query(query_map).hash_stable(&mut hcx, &mut stable_hasher);
            // Prefer entry points which have valid spans for nicer error messages
            // We add an integer to the tuple ensuring that entry points
            // with valid spans are picked first
            let span_cmp = if span == DUMMY_SP { 1 } else { 0 };
            (span_cmp, stable_hasher.finish::<u64>())
        })
        .unwrap()
}

/// Looks for query cycles starting from the last query in `jobs`.
/// If a cycle is found, all queries in the cycle is removed from `jobs` and
/// the function return true.
/// If a cycle was not found, the starting query is removed from `jobs` and
/// the function returns false.
#[cfg(parallel_compiler)]
fn remove_cycle<CTX: QueryContext>(
    query_map: &QueryMap<CTX::DepKind, CTX::Query>,
    jobs: &mut Vec<QueryJobId<CTX::DepKind>>,
    wakelist: &mut Vec<Lrc<QueryWaiter<CTX::DepKind, CTX::Query>>>,
    tcx: CTX,
) -> bool {
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();
    // Look for a cycle starting with the last query in `jobs`
    if let Some(waiter) =
        cycle_check(query_map, jobs.pop().unwrap(), DUMMY_SP, &mut stack, &mut visited)
    {
        // The stack is a vector of pairs of spans and queries; reverse it so that
        // the earlier entries require later entries
        let (mut spans, queries): (Vec<_>, Vec<_>) = stack.into_iter().rev().unzip();

        // Shift the spans so that queries are matched with the span for their waitee
        spans.rotate_right(1);

        // Zip them back together
        let mut stack: Vec<_> = spans.into_iter().zip(queries).collect();

        // Remove the queries in our cycle from the list of jobs to look at
        for r in &stack {
            if let Some(pos) = jobs.iter().position(|j| j == &r.1) {
                jobs.remove(pos);
            }
        }

        // Find the queries in the cycle which are
        // connected to queries outside the cycle
        let entry_points = stack
            .iter()
            .filter_map(|&(span, query)| {
                if query.parent(query_map).is_none() {
                    // This query is connected to the root (it has no query parent)
                    Some((span, query, None))
                } else {
                    let mut waiters = Vec::new();
                    // Find all the direct waiters who lead to the root
                    visit_waiters(query_map, query, |span, waiter| {
                        // Mark all the other queries in the cycle as already visited
                        let mut visited = FxHashSet::from_iter(stack.iter().map(|q| q.1));

                        if connected_to_root(query_map, waiter, &mut visited) {
                            waiters.push((span, waiter));
                        }

                        None
                    });
                    if waiters.is_empty() {
                        None
                    } else {
                        // Deterministically pick one of the waiters to show to the user
                        let waiter = *pick_query(query_map, tcx, &waiters, |s| *s);
                        Some((span, query, Some(waiter)))
                    }
                }
            })
            .collect::<Vec<(Span, QueryJobId<CTX::DepKind>, Option<(Span, QueryJobId<CTX::DepKind>)>)>>();

        // Deterministically pick an entry point
        let (_, entry_point, usage) = pick_query(query_map, tcx, &entry_points, |e| (e.0, e.1));

        // Shift the stack so that our entry point is first
        let entry_point_pos = stack.iter().position(|(_, query)| query == entry_point);
        if let Some(pos) = entry_point_pos {
            stack.rotate_left(pos);
        }

        let usage = usage.as_ref().map(|(span, query)| (*span, query.query(query_map)));

        // Create the cycle error
        let error = CycleError {
            usage,
            cycle: stack
                .iter()
                .map(|&(s, ref q)| QueryInfo { span: s, query: q.query(query_map) })
                .collect(),
        };

        // We unwrap `waiter` here since there must always be one
        // edge which is resumeable / waited using a query latch
        let (waitee_query, waiter_idx) = waiter.unwrap();

        // Extract the waiter we want to resume
        let waiter = waitee_query.latch(query_map).unwrap().extract_waiter(waiter_idx);

        // Set the cycle error so it will be picked up when resumed
        *waiter.cycle.lock() = Some(error);

        // Put the waiter on the list of things to resume
        wakelist.push(waiter);

        true
    } else {
        false
    }
}

/// Detects query cycles by using depth first search over all active query jobs.
/// If a query cycle is found it will break the cycle by finding an edge which
/// uses a query latch and then resuming that waiter.
/// There may be multiple cycles involved in a deadlock, so this searches
/// all active queries for cycles before finally resuming all the waiters at once.
#[cfg(parallel_compiler)]
pub fn deadlock<CTX: QueryContext>(tcx: CTX, registry: &rayon_core::Registry) {
    let on_panic = OnDrop(|| {
        eprintln!("deadlock handler panicked, aborting process");
        process::abort();
    });

    let mut wakelist = Vec::new();
    let query_map = tcx.try_collect_active_jobs().unwrap();
    let mut jobs: Vec<QueryJobId<CTX::DepKind>> = query_map.keys().cloned().collect();

    let mut found_cycle = false;

    while jobs.len() > 0 {
        if remove_cycle(&query_map, &mut jobs, &mut wakelist, tcx) {
            found_cycle = true;
        }
    }

    // Check that a cycle was found. It is possible for a deadlock to occur without
    // a query cycle if a query which can be waited on uses Rayon to do multithreading
    // internally. Such a query (X) may be executing on 2 threads (A and B) and A may
    // wait using Rayon on B. Rayon may then switch to executing another query (Y)
    // which in turn will wait on X causing a deadlock. We have a false dependency from
    // X to Y due to Rayon waiting and a true dependency from Y to X. The algorithm here
    // only considers the true dependency and won't detect a cycle.
    assert!(found_cycle);

    // FIXME: Ensure this won't cause a deadlock before we return
    for waiter in wakelist.into_iter() {
        waiter.notify(registry);
    }

    on_panic.disable();
}
