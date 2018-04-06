// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

use std::mem;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc, Weak};
use rustc_data_structures::OnDrop;
use syntax_pos::Span;
use ty::tls;
use ty::maps::Query;
use ty::maps::plumbing::CycleError;
use ty::context::TyCtxt;
use errors::Diagnostic;
use std::process;
use std::fmt;
use std::collections::HashSet;
#[cfg(parallel_queries)]
use {
    rayon_core,
    parking_lot::{Mutex, Condvar},
    std::sync::atomic::Ordering,
    std::thread,
    std::iter,
    std::iter::FromIterator,
    syntax_pos::DUMMY_SP,
    rustc_data_structures::stable_hasher::{StableHasherResult, StableHasher, HashStable},
};

/// Indicates the state of a query for a given key in a query map
pub(super) enum QueryResult<'tcx> {
    /// An already executing query. The query job can be used to await for its completion
    Started(Lrc<QueryJob<'tcx>>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error / silently panic
    Poisoned,
}

/// A span and a query key
#[derive(Clone, Debug)]
pub struct QueryInfo<'tcx> {
    /// The span for a reason this query was required
    pub span: Span,
    pub query: Query<'tcx>,
}

/// A object representing an active query job.
pub struct QueryJob<'tcx> {
    pub info: QueryInfo<'tcx>,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<Lrc<QueryJob<'tcx>>>,

    /// Diagnostic messages which are emitted while the query executes
    pub diagnostics: Lock<Vec<Diagnostic>>,

    #[cfg(parallel_queries)]
    latch: QueryLatch,
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job
    pub fn new(info: QueryInfo<'tcx>, parent: Option<Lrc<QueryJob<'tcx>>>) -> Self {
        QueryJob {
            diagnostics: Lock::new(Vec::new()),
            info,
            parent,
            #[cfg(parallel_queries)]
            latch: QueryLatch::new(),
        }
    }

    /// Awaits for the query job to complete.
    ///
    /// For single threaded rustc there's no concurrent jobs running, so if we are waiting for any
    /// query that means that there is a query cycle, thus this always running a cycle error.
    pub(super) fn await<'lcx>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'lcx>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        #[cfg(not(parallel_queries))]
        {
            self.find_cycle_in_stack(tcx, span)
        }

        #[cfg(parallel_queries)]
        {
            tls::with_related_context(tcx, move |icx| {
                let mut waiter = QueryWaiter {
                    query: &icx.query,
                    span,
                    cycle: None,
                    condvar: Condvar::new(),
                };
                self.latch.await(&mut waiter);

                match waiter.cycle {
                    None => Ok(()),
                    Some(cycle) => Err(cycle)
                }
            })
        }
    }

    #[cfg(not(parallel_queries))]
    fn find_cycle_in_stack<'lcx>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'lcx>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        // Get the current executing query (waiter) and find the waitee amongst its parents
        let mut current_job = tls::with_related_context(tcx, |icx| icx.query.clone());
        let mut cycle = Vec::new();

        while let Some(job) = current_job {
            cycle.insert(0, job.info.clone());

            if &*job as *const _ == self as *const _ {
                // This is the end of the cycle
                // The span entry we included was for the usage
                // of the cycle itself, and not part of the cycle
                // Replace it with the span which caused the cycle to form
                cycle[0].span = span;
                // Find out why the cycle itself was used
                let usage = job.parent.as_ref().map(|parent| {
                    (job.info.span, parent.info.query.clone())
                });
                return Err(CycleError { usage, cycle });
            }

            current_job = job.parent.clone();
        }

        panic!("did not find a cycle")
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    pub fn signal_complete(&self, tcx: TyCtxt<'_, 'tcx, '_>) {
        #[cfg(parallel_queries)]
        self.latch.set(tcx);
    }
}

#[cfg(parallel_queries)]
struct QueryWaiter<'a, 'tcx: 'a> {
    query: &'a Option<Lrc<QueryJob<'tcx>>>,
    condvar: Condvar,
    span: Span,
    cycle: Option<CycleError<'tcx>>,
}

#[cfg(parallel_queries)]
impl<'a, 'tcx> QueryWaiter<'a, 'tcx> {
    fn notify(&self, tcx: TyCtxt<'_, '_, '_>, registry: &rayon_core::Registry) {
        rayon_core::mark_unblocked(registry);
        self.condvar.notify_one();
    }
}

#[cfg(parallel_queries)]
struct QueryLatchInfo {
    complete: bool,
    waiters: Vec<&'static mut QueryWaiter<'static, 'static>>,
}

#[cfg(parallel_queries)]
struct QueryLatch {
    info: Mutex<QueryLatchInfo>,
}

#[cfg(parallel_queries)]
impl QueryLatch {
    fn new() -> Self {
        QueryLatch {
            info: Mutex::new(QueryLatchInfo {
                complete: false,
                waiters: Vec::new(),
            }),
        }
    }

    fn await(&self, waiter: &mut QueryWaiter<'_, '_>) {
        let mut info = self.info.lock();
        if !info.complete {
            let waiter = &*waiter;
            unsafe {
                #[allow(mutable_transmutes)]
                info.waiters.push(mem::transmute(waiter));
            }
            // If this detects a deadlock and the deadlock handler want to resume this thread
            // we have to be in the `wait` call. This is ensured by the deadlock handler
            // getting the self.info lock.
            rayon_core::mark_blocked();
            waiter.condvar.wait(&mut info);
        }
    }

    fn set(&self, tcx: TyCtxt<'_, '_, '_>) {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        info.complete = true;
        let registry = rayon_core::Registry::current();
        for waiter in info.waiters.drain(..) {
            waiter.notify(tcx, &registry);
        }
    }

    fn resume_waiter(
        &self,
        waiter: usize,
        error: CycleError
    ) -> &'static mut QueryWaiter<'static, 'static> {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        // Remove the waiter from the list of waiters
        let waiter = info.waiters.remove(waiter);

        // Set the cycle error it will be picked it up when resumed
        waiter.cycle = unsafe { Some(mem::transmute(error)) };

        waiter
    }
}

#[cfg(parallel_queries)]
type Ref<'tcx> = *const QueryJob<'tcx>;

#[cfg(parallel_queries)]
type Waiter<'tcx> = (Ref<'tcx>, usize);

#[cfg(parallel_queries)]
fn visit_waiters<'tcx, F>(query_ref: Ref<'tcx>, mut visit: F) -> Option<Option<Waiter<'tcx>>>
where
    F: FnMut(Span, Ref<'tcx>) -> Option<Option<Waiter<'tcx>>>
{
    let query = unsafe { &*query_ref };
    if let Some(ref parent) = query.parent {
        if let Some(cycle) = visit(query.info.span, &**parent as Ref) {
            return Some(cycle);
        }
    }
    for (i, waiter) in query.latch.info.lock().waiters.iter().enumerate() {
        if let Some(ref waiter_query) = waiter.query {
            if visit(waiter.span, &**waiter_query as Ref).is_some() {
                return Some(Some((query_ref, i)));
            }
        }
    }
    None
}

#[cfg(parallel_queries)]
fn cycle_check<'tcx>(query: Ref<'tcx>,
                     span: Span,
                     stack: &mut Vec<(Span, Ref<'tcx>)>,
                     visited: &mut HashSet<Ref<'tcx>>) -> Option<Option<Waiter<'tcx>>> {
    if visited.contains(&query) {
        return if let Some(p) = stack.iter().position(|q| q.1 == query) {
            // Remove previous stack entries
            stack.splice(0..p, iter::empty());
            // Replace the span for the first query with the cycle cause
            stack[0].0 = span;
            Some(None)
        } else {
            None
        }
    }

    visited.insert(query);
    stack.push((span, query));

    let r = visit_waiters(query, |span, successor| {
        cycle_check(successor, span, stack, visited)
    });

    if r.is_none() {
        stack.pop();
    }

    r
}

#[cfg(parallel_queries)]
fn connected_to_root<'tcx>(query: Ref<'tcx>, visited: &mut HashSet<Ref<'tcx>>) -> bool {
    if visited.contains(&query) {
        return false;
    }

    if unsafe { (*query).parent.is_none() } {
        return true;
    }

    visited.insert(query);

    let mut connected = false;

    visit_waiters(query, |_, successor| {
        if connected_to_root(successor, visited) {
            Some(None)
        } else {
            None
        }
    }).is_some()
}

#[cfg(parallel_queries)]
fn query_entry<'tcx>(r: Ref<'tcx>) -> QueryInfo<'tcx> {
    unsafe { (*r).info.clone() }
}

#[cfg(parallel_queries)]
fn remove_cycle<'tcx>(
    jobs: &mut Vec<Ref<'tcx>>,
    wakelist: &mut Vec<&'static mut QueryWaiter<'static, 'static>>,
    tcx: TyCtxt<'_, 'tcx, '_>
) {
    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    if let Some(waiter) = cycle_check(jobs.pop().unwrap(),
                                      DUMMY_SP,
                                      &mut stack,
                                      &mut visited) {
        // Reverse the stack so earlier entries require later entries
        stack.reverse();

        let mut spans: Vec<_> = stack.iter().map(|e| e.0).collect();
        let queries = stack.iter().map(|e| e.1);

        // Shift the spans so that a query is matched the span for its waitee
        let last = spans.pop().unwrap();
        spans.insert(0, last);

        let mut stack: Vec<_> = spans.into_iter().zip(queries).collect();

        // Remove the queries in our cycle from the list of jobs to look at
        for r in &stack {
            jobs.remove_item(&r.1);
        }

        let (waitee_query, waiter_idx) = waiter.unwrap();
        let waitee_query = unsafe { &*waitee_query };

        // Find the queries in the cycle which are
        // connected to queries outside the cycle
        let entry_points: Vec<Ref<'_>> = stack.iter().filter_map(|query| {
            // Mark all the other queries in the cycle as already visited
            let mut visited = HashSet::from_iter(stack.iter().filter_map(|q| {
                if q.1 != query.1 {
                    Some(q.1)
                } else {
                    None
                }
            }));

            if connected_to_root(query.1, &mut visited) {
                Some(query.1)
            } else {
                None
            }
        }).collect();

        // Deterministically pick an entry point
        // FIXME: Sort this instead
        let mut hcx = tcx.create_stable_hashing_context();
        let entry_point = *entry_points.iter().min_by_key(|&&q| {
            let mut stable_hasher = StableHasher::<u64>::new();
            unsafe { (*q).info.query.hash_stable(&mut hcx, &mut stable_hasher); }
            stable_hasher.finish()
        }).unwrap();

        // Shift the stack until our entry point is first
        while stack[0].1 != entry_point {
            let last = stack.pop().unwrap();
            stack.insert(0, last);
        }

        let mut error = CycleError {
            usage: None,
            cycle: stack.iter().map(|&(s, q)| QueryInfo {
                span: s,
                query: unsafe { (*q).info.query.clone() },
            } ).collect(),
        };

        wakelist.push(waitee_query.latch.resume_waiter(waiter_idx, error));
    }
}

#[cfg(parallel_queries)]
pub fn handle_deadlock() {
    use syntax;
    use syntax_pos;

    let registry = rayon_core::Registry::current();

    let gcx_ptr = tls::GCX_PTR.with(|gcx_ptr| {
        gcx_ptr as *const _
    });
    let gcx_ptr = unsafe { &*gcx_ptr };

    let syntax_globals = syntax::GLOBALS.with(|syntax_globals| {
        syntax_globals as *const _
    });
    let syntax_globals = unsafe { &*syntax_globals };

    let syntax_pos_globals = syntax_pos::GLOBALS.with(|syntax_pos_globals| {
        syntax_pos_globals as *const _
    });
    let syntax_pos_globals = unsafe { &*syntax_pos_globals };
    thread::spawn(move || {
        tls::GCX_PTR.set(gcx_ptr, || {
            syntax_pos::GLOBALS.set(syntax_pos_globals, || {
                syntax_pos::GLOBALS.set(syntax_pos_globals, || {
                    tls::with_thread_locals(|| {
                        unsafe {
                            tls::with_global(|tcx| deadlock(tcx, &registry))
                        }
                    })
                })
            })
        })
    });
}

#[cfg(parallel_queries)]
fn deadlock(tcx: TyCtxt<'_, '_, '_>, registry: &rayon_core::Registry) {
    let on_panic = OnDrop(|| {
        eprintln!("deadlock handler panicked, aborting process");
        process::abort();
    });

    let mut wakelist = Vec::new();
    let mut jobs: Vec<_> = tcx.maps.collect_active_jobs().iter().map(|j| &**j as Ref).collect();

    while jobs.len() > 0 {
        remove_cycle(&mut jobs, &mut wakelist, tcx);
    }

    // FIXME: Panic if no cycle is detected

    // FIXME: Write down the conditions when a deadlock happens without a cycle

    // FIXME: Ensure this won't cause a deadlock before we return
    for waiter in wakelist.into_iter() {
        waiter.notify(tcx, registry);
    }

    mem::forget(on_panic);
}
