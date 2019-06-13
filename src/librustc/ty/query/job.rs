#![allow(warnings)]

use std::mem;
use std::process;
use std::{fmt, ptr};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc, Weak};
use rustc_data_structures::OnDrop;
use rustc_data_structures::jobserver;
use syntax_pos::Span;

use crate::ty::tls;
use crate::ty::query::Query;
use crate::ty::query::plumbing::CycleError;
#[cfg(not(parallel_compiler))]
use crate::ty::query::{
    plumbing::TryGetJob,
    config::QueryDescription,
};
use crate::ty::context::TyCtxt;

#[cfg(parallel_compiler)]
use {
    rustc_rayon_core as rayon_core,
    parking_lot::{Mutex, Condvar},
    std::sync::atomic::Ordering,
    std::thread,
    std::iter,
    std::iter::FromIterator,
    syntax_pos::DUMMY_SP,
    rustc_data_structures::stable_hasher::{StableHasherResult, StableHasher, HashStable},
};

/// Indicates the state of a query for a given key in a query map.
pub(super) enum QueryResult<'tcx> {
    /// An already executing query. The query job can be used to await for its completion.
    Started(Lrc<QueryJob<'tcx>>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error or
    /// silently panic.
    Poisoned,
}

/// Represents a span and a query key.
#[derive(Clone, Debug)]
pub struct QueryInfo<'tcx> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,
    pub query: Query<'tcx>,
}

/// Representss an object representing an active query job.
pub struct QueryJob<'tcx> {
    pub info: QueryInfo<'tcx>,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<Lrc<QueryJob<'tcx>>>,

    /// The latch that is used to wait on this job.
    #[cfg(parallel_compiler)]
    latch: QueryLatch<'tcx>,
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job.
    pub fn new(info: QueryInfo<'tcx>, parent: Option<Lrc<QueryJob<'tcx>>>) -> Self {
        QueryJob {
            info,
            parent,
            #[cfg(parallel_compiler)]
            latch: QueryLatch::new(),
        }
    }

    /// Awaits for the query job to complete.
    #[cfg(parallel_compiler)]
    pub(super) fn r#await(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        tls::with_related_context(tcx, move |icx| {
            let mut waiter = Lrc::new(QueryWaiter {
                query: icx.query.clone(),
                span,
                cycle: Lock::new(None),
                condvar: Condvar::new(),
            });
            self.latch.r#await(&waiter);
            // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
            // although another thread may still have a Lrc reference so we cannot
            // use Lrc::get_mut
            let mut cycle = waiter.cycle.lock();
            match cycle.take() {
                None => Ok(()),
                Some(cycle) => Err(cycle)
            }
        })
    }

    #[cfg(not(parallel_compiler))]
    pub(super) fn find_cycle_in_stack(&self, tcx: TyCtxt<'tcx>, span: Span) -> CycleError<'tcx> {
        // Get the current executing query (waiter) and find the waitee amongst its parents
        let mut current_job = tls::with_related_context(tcx, |icx| icx.query.clone());
        let mut cycle = Vec::new();

        while let Some(job) = current_job {
            cycle.push(job.info.clone());

            if ptr::eq(&*job, self) {
                cycle.reverse();

                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = job.parent.as_ref().map(|parent| {
                    (job.info.span, parent.info.query.clone())
                });
                return CycleError { usage, cycle };
            }

            current_job = job.parent.clone();
        }

        panic!("did not find a cycle")
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    pub fn signal_complete(&self) {
        #[cfg(parallel_compiler)]
        self.latch.set();
    }

    fn as_ptr(&self) -> *const QueryJob<'tcx> {
        self as *const _
    }
}

#[cfg(parallel_compiler)]
struct QueryWaiter<'tcx> {
    query: Option<Lrc<QueryJob<'tcx>>>,
    condvar: Condvar,
    span: Span,
    cycle: Lock<Option<CycleError<'tcx>>>,
}

#[cfg(parallel_compiler)]
impl<'tcx> QueryWaiter<'tcx> {
    fn notify(&self, registry: &rayon_core::Registry) {
        rayon_core::mark_unblocked(registry);
        self.condvar.notify_one();
    }
}

#[cfg(parallel_compiler)]
struct QueryLatchInfo<'tcx> {
    complete: bool,
    waiters: Vec<Lrc<QueryWaiter<'tcx>>>,
}

#[cfg(parallel_compiler)]
struct QueryLatch<'tcx> {
    info: Mutex<QueryLatchInfo<'tcx>>,
}

#[cfg(parallel_compiler)]
impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch {
            info: Mutex::new(QueryLatchInfo {
                complete: false,
                waiters: Vec::new(),
            }),
        }
    }

    /// Awaits the caller on this latch by blocking the current thread.
    fn r#await(&self, waiter: &Lrc<QueryWaiter<'tcx>>) {
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
    fn extract_waiter(
        &self,
        waiter: usize,
    ) -> Lrc<QueryWaiter<'tcx>> {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        // Remove the waiter from the list of waiters
        info.waiters.remove(waiter)
    }
}

/// A resumable waiter of a query. The usize is the index into waiters in the query's latch
#[cfg(parallel_compiler)]
type Waiter<'tcx> = (Lrc<QueryJob<'tcx>>, usize);

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
fn visit_waiters<'tcx, F>(query: Lrc<QueryJob<'tcx>>, mut visit: F) -> Option<Option<Waiter<'tcx>>>
where
    F: FnMut(Span, Lrc<QueryJob<'tcx>>) -> Option<Option<Waiter<'tcx>>>
{
    // Visit the parent query which is a non-resumable waiter since it's on the same stack
    if let Some(ref parent) = query.parent {
        if let Some(cycle) = visit(query.info.span, parent.clone()) {
            return Some(cycle);
        }
    }

    // Visit the explicit waiters which use condvars and are resumable
    for (i, waiter) in query.latch.info.lock().waiters.iter().enumerate() {
        if let Some(ref waiter_query) = waiter.query {
            if visit(waiter.span, waiter_query.clone()).is_some() {
                // Return a value which indicates that this waiter can be resumed
                return Some(Some((query.clone(), i)));
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
fn cycle_check<'tcx>(query: Lrc<QueryJob<'tcx>>,
                     span: Span,
                     stack: &mut Vec<(Span, Lrc<QueryJob<'tcx>>)>,
                     visited: &mut FxHashSet<*const QueryJob<'tcx>>
) -> Option<Option<Waiter<'tcx>>> {
    if !visited.insert(query.as_ptr()) {
        return if let Some(p) = stack.iter().position(|q| q.1.as_ptr() == query.as_ptr()) {
            // We detected a query cycle, fix up the initial span and return Some

            // Remove previous stack entries
            stack.drain(0..p);
            // Replace the span for the first query with the cycle cause
            stack[0].0 = span;
            Some(None)
        } else {
            None
        }
    }

    // Query marked as visited is added it to the stack
    stack.push((span, query.clone()));

    // Visit all the waiters
    let r = visit_waiters(query, |span, successor| {
        cycle_check(successor, span, stack, visited)
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
fn connected_to_root<'tcx>(
    query: Lrc<QueryJob<'tcx>>,
    visited: &mut FxHashSet<*const QueryJob<'tcx>>
) -> bool {
    // We already visited this or we're deliberately ignoring it
    if !visited.insert(query.as_ptr()) {
        return false;
    }

    // This query is connected to the root (it has no query parent), return true
    if query.parent.is_none() {
        return true;
    }

    visit_waiters(query, |_, successor| {
        if connected_to_root(successor, visited) {
            Some(None)
        } else {
            None
        }
    }).is_some()
}

// Deterministically pick an query from a list
#[cfg(parallel_compiler)]
fn pick_query<'a, 'tcx, T, F: Fn(&T) -> (Span, Lrc<QueryJob<'tcx>>)>(
    tcx: TyCtxt<'tcx>,
    queries: &'a [T],
    f: F,
) -> &'a T {
    // Deterministically pick an entry point
    // FIXME: Sort this instead
    let mut hcx = tcx.create_stable_hashing_context();
    queries.iter().min_by_key(|v| {
        let (span, query) = f(v);
        let mut stable_hasher = StableHasher::<u64>::new();
        query.info.query.hash_stable(&mut hcx, &mut stable_hasher);
        // Prefer entry points which have valid spans for nicer error messages
        // We add an integer to the tuple ensuring that entry points
        // with valid spans are picked first
        let span_cmp = if span == DUMMY_SP { 1 } else { 0 };
        (span_cmp, stable_hasher.finish())
    }).unwrap()
}

/// Looks for query cycles starting from the last query in `jobs`.
/// If a cycle is found, all queries in the cycle is removed from `jobs` and
/// the function return true.
/// If a cycle was not found, the starting query is removed from `jobs` and
/// the function returns false.
#[cfg(parallel_compiler)]
fn remove_cycle<'tcx>(
    jobs: &mut Vec<Lrc<QueryJob<'tcx>>>,
    wakelist: &mut Vec<Lrc<QueryWaiter<'tcx>>>,
    tcx: TyCtxt<'tcx>,
) -> bool {
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();
    // Look for a cycle starting with the last query in `jobs`
    if let Some(waiter) = cycle_check(jobs.pop().unwrap(),
                                      DUMMY_SP,
                                      &mut stack,
                                      &mut visited) {
        // The stack is a vector of pairs of spans and queries; reverse it so that
        // the earlier entries require later entries
        let (mut spans, queries): (Vec<_>, Vec<_>) = stack.into_iter().rev().unzip();

        // Shift the spans so that queries are matched with the span for their waitee
        spans.rotate_right(1);

        // Zip them back together
        let mut stack: Vec<_> = spans.into_iter().zip(queries).collect();

        // Remove the queries in our cycle from the list of jobs to look at
        for r in &stack {
            if let Some(pos) = jobs.iter().position(|j| j.as_ptr() == r.1.as_ptr()) {
                jobs.remove(pos);
            }
        }

        // Find the queries in the cycle which are
        // connected to queries outside the cycle
        let entry_points = stack.iter().filter_map(|(span, query)| {
            if query.parent.is_none() {
                // This query is connected to the root (it has no query parent)
                Some((*span, query.clone(), None))
            } else {
                let mut waiters = Vec::new();
                // Find all the direct waiters who lead to the root
                visit_waiters(query.clone(), |span, waiter| {
                    // Mark all the other queries in the cycle as already visited
                    let mut visited = FxHashSet::from_iter(stack.iter().map(|q| q.1.as_ptr()));

                    if connected_to_root(waiter.clone(), &mut visited) {
                        waiters.push((span, waiter));
                    }

                    None
                });
                if waiters.is_empty() {
                    None
                } else {
                    // Deterministically pick one of the waiters to show to the user
                    let waiter = pick_query(tcx, &waiters, |s| s.clone()).clone();
                    Some((*span, query.clone(), Some(waiter)))
                }
            }
        }).collect::<Vec<(Span, Lrc<QueryJob<'tcx>>, Option<(Span, Lrc<QueryJob<'tcx>>)>)>>();

        // Deterministically pick an entry point
        let (_, entry_point, usage) = pick_query(tcx, &entry_points, |e| (e.0, e.1.clone()));

        // Shift the stack so that our entry point is first
        let entry_point_pos = stack.iter().position(|(_, query)| {
            query.as_ptr() == entry_point.as_ptr()
        });
        if let Some(pos) = entry_point_pos {
            stack.rotate_left(pos);
        }

        let usage = usage.as_ref().map(|(span, query)| (*span, query.info.query.clone()));

        // Create the cycle error
        let mut error = CycleError {
            usage,
            cycle: stack.iter().map(|&(s, ref q)| QueryInfo {
                span: s,
                query: q.info.query.clone(),
            } ).collect(),
        };

        // We unwrap `waiter` here since there must always be one
        // edge which is resumeable / waited using a query latch
        let (waitee_query, waiter_idx) = waiter.unwrap();

        // Extract the waiter we want to resume
        let waiter = waitee_query.latch.extract_waiter(waiter_idx);

        // Set the cycle error so it will be picked up when resumed
        *waiter.cycle.lock() = Some(error);

        // Put the waiter on the list of things to resume
        wakelist.push(waiter);

        true
    } else {
        false
    }
}

/// Creates a new thread and forwards information in thread locals to it.
/// The new thread runs the deadlock handler.
/// Must only be called when a deadlock is about to happen.
#[cfg(parallel_compiler)]
pub unsafe fn handle_deadlock() {
    use syntax;
    use syntax_pos;

    let registry = rayon_core::Registry::current();

    let gcx_ptr = tls::GCX_PTR.with(|gcx_ptr| {
        gcx_ptr as *const _
    });
    let gcx_ptr = &*gcx_ptr;

    let syntax_globals = syntax::GLOBALS.with(|syntax_globals| {
        syntax_globals as *const _
    });
    let syntax_globals = &*syntax_globals;

    let syntax_pos_globals = syntax_pos::GLOBALS.with(|syntax_pos_globals| {
        syntax_pos_globals as *const _
    });
    let syntax_pos_globals = &*syntax_pos_globals;
    thread::spawn(move || {
        tls::GCX_PTR.set(gcx_ptr, || {
            syntax_pos::GLOBALS.set(syntax_pos_globals, || {
                syntax_pos::GLOBALS.set(syntax_pos_globals, || {
                    tls::with_thread_locals(|| {
                        tls::with_global(|tcx| deadlock(tcx, &registry))
                    })
                })
            })
        })
    });
}

/// Detects query cycles by using depth first search over all active query jobs.
/// If a query cycle is found it will break the cycle by finding an edge which
/// uses a query latch and then resuming that waiter.
/// There may be multiple cycles involved in a deadlock, so this searches
/// all active queries for cycles before finally resuming all the waiters at once.
#[cfg(parallel_compiler)]
fn deadlock(tcx: TyCtxt<'_>, registry: &rayon_core::Registry) {
    let on_panic = OnDrop(|| {
        eprintln!("deadlock handler panicked, aborting process");
        process::abort();
    });

    let mut wakelist = Vec::new();
    let mut jobs: Vec<_> = tcx.queries.collect_active_jobs();

    let mut found_cycle = false;

    while jobs.len() > 0 {
        if remove_cycle(&mut jobs, &mut wakelist, tcx) {
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
