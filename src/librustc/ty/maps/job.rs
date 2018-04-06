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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc, Weak};
use rustc_data_structures::OnDrop;
use rayon_core::registry::{self, Registry, WorkerThread};
use rayon_core::fiber::{Fiber, Waitable, WaiterLatch};
use rayon_core::latch::{LatchProbe, Latch};
use syntax_pos::Span;
use ty::tls;
use ty::maps::Query;
use ty::maps::plumbing::CycleError;
use ty::context::TyCtxt;
use errors::Diagnostic;
use std::process;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::collections::HashSet;
#[cfg(parallel_queries)]
use {
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

    latch: Lock<QueryLatch>,
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job
    pub fn new(info: QueryInfo<'tcx>, parent: Option<Lrc<QueryJob<'tcx>>>) -> Self {
        QueryJob {
            diagnostics: Lock::new(Vec::new()),
            info,
            parent,
            latch: Lock::new(QueryLatch::new()),
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
                let cycle = (span, Lock::new(None));

                {
                    let icx = tls::ImplicitCtxt {
                        waiter_cycle: Some(&cycle),
                        ..icx.clone()
                    };

                    tls::enter_context(&icx, |_| {
                        registry::in_worker(|worker, _| {
                            unsafe {
                                worker.wait_enqueue(self);
                            }
                        });
                    })
                }

                match cycle.1.into_inner() {
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
    pub fn signal_complete(&self) {
        #[cfg(parallel_queries)]
        self.latch.lock().set();
    }
}

#[cfg(parallel_queries)]
impl<'tcx> LatchProbe for QueryJob<'tcx> {
    #[inline]
    fn probe(&self) -> bool {
        self.latch.lock().complete
    }
}

#[cfg(parallel_queries)]
impl<'tcx> Latch for QueryJob<'tcx> {
    fn set(&self) {
        self.latch.lock().set();
    }
}

#[cfg(parallel_queries)]
impl<'tcx> Waitable for QueryJob<'tcx> {
    fn complete(&self, _worker_thread: &WorkerThread) -> bool {
        self.probe()
    }

    fn await(&self, worker_thread: &WorkerThread, waiter: Fiber, tlv: usize) {
        let mut latch = self.latch.lock();
        if latch.complete {
            worker_thread.registry.resume_fiber(worker_thread.index(), waiter);
        } else {
            latch.waiters.push(QueryWaiter {
                worker_index: worker_thread.index(),
                fiber: waiter,
                tlv,
            });
        }
    }
}

#[cfg(parallel_queries)]
struct QueryWaiter {
    worker_index: usize,
    fiber: Fiber,
    tlv: usize,
}

#[cfg(parallel_queries)]
impl QueryWaiter {
    fn icx<'a, 'b, 'gcx, 'tcx>(&'a self) -> *const tls::ImplicitCtxt<'b, 'gcx, 'tcx> {
        self.tlv as *const tls::ImplicitCtxt
    }
}

#[cfg(parallel_queries)]
struct QueryLatch {
    complete: bool,
    waiters: Vec<QueryWaiter>,
}

#[cfg(parallel_queries)]
impl QueryLatch {
    fn new() -> Self {
        QueryLatch {
            complete: false,
            waiters: Vec::new(),
        }
    }

    fn set(&mut self) {
        debug_assert!(!self.complete);
        self.complete = true;
        if !self.waiters.is_empty() {
            let registry = Registry::current();
            for waiter in self.waiters.drain(..) {
                registry.resume_fiber(waiter.worker_index, waiter.fiber);
            }
            registry.signal();
        }
    }

    fn resume_waiter(&mut self, waiter: usize, error: CycleError) {
        debug_assert!(!self.complete);
        // Remove the waiter from the list of waiters
        let waiter = self.waiters.remove(waiter);

        // Set the cycle error in its icx so it can pick it up when resumed
        {
            let icx = unsafe { &*waiter.icx() };
            *icx.waiter_cycle.unwrap().1.lock() = Some(error);
        }

        // Resume the waiter
        let registry = Registry::current();
        registry.resume_fiber(waiter.worker_index, waiter.fiber);
    }
}

fn print_job<'a, 'tcx, 'lcx>(tcx: TyCtxt<'a, 'tcx, 'lcx>, job: &QueryJob<'tcx>) -> String {
    format!("[{}] {:x} {:?}",
        0/*entry.id*/, job as *const _ as usize, job.info.query.describe(tcx))
}

type Ref<'tcx> = *const QueryJob<'tcx>;

type Waiter<'tcx> = (Ref<'tcx>, usize);

fn visit_waiters<'tcx, F>(query_ref: Ref<'tcx>, mut visit: F) -> Option<Option<Waiter<'tcx>>>
where
    F: FnMut(Span, Ref<'tcx>) -> Option<Option<Waiter<'tcx>>>
{
    let query = unsafe { &*query_ref };
    if let Some(ref parent) = query.parent {
        //eprintln!("visiting parent {:?} of query {:?}", parent, query_ref);
        if let Some(cycle) = visit(query.info.span, &**parent as Ref) {
            return Some(cycle);
        }
    }
    for (i, waiter) in query.latch.lock().waiters.iter().enumerate() {
        let icx = unsafe { &*waiter.icx() };
        if let Some(ref waiter_query) = icx.query {
            //eprintln!("visiting waiter {:?} of query {:?}", waiter, query_ref);
            if visit(icx.waiter_cycle.unwrap().0, &**waiter_query as Ref).is_some() {
                // We found a cycle, return this edge as the waiter
                return Some(Some((query_ref, i)));
            }
        }
    }
    None
}

fn cycle_check<'tcx>(query: Ref<'tcx>,
                     span: Span,
                     stack: &mut Vec<(Span, Ref<'tcx>)>,
                     visited: &mut HashSet<Ref<'tcx>>) -> Option<Option<Waiter<'tcx>>> {
    if visited.contains(&query) {
        //eprintln!("visited query {:?} already for cycle {:#?}", query, stack);

        return if let Some(p) = stack.iter().position(|q| q.1 == query) {
            // Remove previous stack entries
            stack.splice(0..p, iter::empty());
            // Replace the span for the first query with the cycle cause
            stack[0].0 = span;
        //eprintln!("[found on stack] visited query {:?} already for cycle {:#?}", query, stack);
            Some(None)
        } else {
        /*eprintln!("[not found on stack] visited query {:?} already for cycle {:#?}",
            query, stack);*/
            None
        }
    }

    //eprintln!("looking for cycle {:#?} in query {:?}", stack, query);

    visited.insert(query);
    stack.push((span, query));

    let r = visit_waiters(query, |span, successor| {
        //eprintln!("found successor {:?} in query {:?}", successor, query);
        cycle_check(successor, span, stack, visited)
    });

    //eprintln!("result for query {:?} {:?}", query, r);

    if r.is_none() {
        stack.pop();
    }

    r
}

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

fn query_entry<'tcx>(r: Ref<'tcx>) -> QueryInfo<'tcx> {
    unsafe { (*r).info.clone() }
}

fn remove_cycle<'tcx>(jobs: &mut Vec<Ref<'tcx>>, tcx: TyCtxt<'_, 'tcx, '_>) {
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

/*        eprintln!("found cycle {:#?} with waitee {:?}", stack, waitee_query.info.query);

        for r in &stack { unsafe {
            eprintln!("- query: {}", (*r.1).info.query.describe(tcx));
        } }
*/
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

        //eprintln!("entry_points {:#?}", entry_points);

        // Deterministically pick an entry point
        // FIXME: Sort this instead
        let mut hcx = tcx.create_stable_hashing_context();
        let entry_point = *entry_points.iter().min_by_key(|&&q| {
            let mut stable_hasher = StableHasher::<u64>::new();
            unsafe { (*q).info.query.hash_stable(&mut hcx, &mut stable_hasher); }
            stable_hasher.finish()
        }).unwrap();

        /*unsafe {
            eprintln!("found entry point {:?}  {:?}",
                        entry_point, (*entry_point).info.query);
        }*/

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

        waitee_query.latch.lock().resume_waiter(waiter_idx, error);
    }
}

pub fn deadlock() {
    let on_panic = OnDrop(|| {
        eprintln!("deadlock handler panicked, aborting process");
        process::abort();
    });

    //eprintln!("saw rayon deadlock");
    unsafe { tls::with_global_query(|tcx| {
        let mut jobs: Vec<_> = tcx.maps.collect_active_jobs().iter().map(|j| &**j as Ref).collect();
/*
        for job in &jobs { unsafe {
            eprintln!("still active query: {}", print_job(tcx, &**job));
            if let Some(ref parent) = (**job).parent {
                eprintln!("   - has parent: {}", print_job(tcx, &**parent));
            }
            for (i, waiter) in (**job).latch.lock().waiters.iter().enumerate() {
                let icx = &*waiter.icx();
                if let Some(ref query) = icx.query {
                    eprintln!("   - has waiter d{}: {}", i, print_job(tcx, &**query));

                } else {
                    eprintln!("   - has no-query waiter d{}", i);
                }
            }
        } }
*/
        while jobs.len() > 0 {
            remove_cycle(&mut jobs, tcx);
        }
    })};
    //eprintln!("aborting due to deadlock");
    //process::abort();
    mem::forget(on_panic);
    Registry::current().signal();
}
