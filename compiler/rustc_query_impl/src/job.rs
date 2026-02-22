use std::io::Write;
use std::iter;
use std::ops::ControlFlow;
use std::sync::Arc;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_middle::query::{
    CycleError, QueryInfo, QueryJob, QueryJobId, QueryLatch, QueryStackDeferred, QueryStackFrame,
    QueryWaiter,
};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::{DUMMY_SP, Span};

use crate::plumbing::collect_active_jobs_from_all_queries;

/// Map from query job IDs to job information collected by
/// `collect_active_jobs_from_all_queries`.
#[derive(Debug, Default)]
pub struct QueryJobMap<'tcx> {
    map: FxHashMap<QueryJobId, QueryJobInfo<'tcx>>,
}

impl<'tcx> QueryJobMap<'tcx> {
    /// Adds information about a job ID to the job map.
    ///
    /// Should only be called by `gather_active_jobs_inner`.
    pub(crate) fn insert(&mut self, id: QueryJobId, info: QueryJobInfo<'tcx>) {
        self.map.insert(id, info);
    }

    fn frame_of(&self, id: QueryJobId) -> &QueryStackFrame<QueryStackDeferred<'tcx>> {
        &self.map[&id].frame
    }

    fn span_of(&self, id: QueryJobId) -> Span {
        self.map[&id].job.span
    }

    fn parent_of(&self, id: QueryJobId) -> Option<QueryJobId> {
        self.map[&id].job.parent
    }

    fn latch_of(&self, id: QueryJobId) -> Option<&QueryLatch<'tcx>> {
        self.map[&id].job.latch.as_ref()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct QueryJobInfo<'tcx> {
    pub(crate) frame: QueryStackFrame<QueryStackDeferred<'tcx>>,
    pub(crate) job: QueryJob<'tcx>,
}

pub(crate) fn find_cycle_in_stack<'tcx>(
    id: QueryJobId,
    job_map: QueryJobMap<'tcx>,
    current_job: &Option<QueryJobId>,
    span: Span,
) -> CycleError<QueryStackDeferred<'tcx>> {
    // Find the waitee amongst `current_job` parents
    let mut cycle = Vec::new();
    let mut current_job = Option::clone(current_job);

    while let Some(job) = current_job {
        let info = &job_map.map[&job];
        cycle.push(QueryInfo { span: info.job.span, frame: info.frame.clone() });

        if job == id {
            cycle.reverse();

            // This is the end of the cycle
            // The span entry we included was for the usage
            // of the cycle itself, and not part of the cycle
            // Replace it with the span which caused the cycle to form
            cycle[0].span = span;
            // Find out why the cycle itself was used
            let usage = try {
                let parent = info.job.parent?;
                (info.job.span, job_map.frame_of(parent).clone())
            };
            return CycleError { usage, cycle };
        }

        current_job = info.job.parent;
    }

    panic!("did not find a cycle")
}

#[cold]
#[inline(never)]
pub(crate) fn find_dep_kind_root<'tcx>(
    id: QueryJobId,
    job_map: QueryJobMap<'tcx>,
) -> (QueryJobInfo<'tcx>, usize) {
    let mut depth = 1;
    let info = &job_map.map[&id];
    let dep_kind = info.frame.dep_kind;
    let mut current_id = info.job.parent;
    let mut last_layout = (info.clone(), depth);

    while let Some(id) = current_id {
        let info = &job_map.map[&id];
        if info.frame.dep_kind == dep_kind {
            depth += 1;
            last_layout = (info.clone(), depth);
        }
        current_id = info.job.parent;
    }
    last_layout
}

/// A resumable waiter of a query. The usize is the index into waiters in the query's latch
type Waiter = (QueryJobId, usize);

/// Visits all the non-resumable and resumable waiters of a query.
/// Only waiters in a query are visited.
/// `visit` is called for every waiter and is passed a query waiting on `query`
/// and a span indicating the reason the query waited on `query`.
/// If `visit` returns `Break`, this function also returns `Break`,
/// and if all `visit` calls returns `Continue` it also returns `Continue`.
/// For visits of non-resumable waiters it returns the return value of `visit`.
/// For visits of resumable waiters it returns information required to resume that waiter.
fn visit_waiters<'tcx>(
    job_map: &QueryJobMap<'tcx>,
    query: QueryJobId,
    mut visit: impl FnMut(Span, QueryJobId) -> ControlFlow<Option<Waiter>>,
) -> ControlFlow<Option<Waiter>> {
    // Visit the parent query which is a non-resumable waiter since it's on the same stack
    if let Some(parent) = job_map.parent_of(query) {
        visit(job_map.span_of(query), parent)?;
    }

    // Visit the explicit waiters which use condvars and are resumable
    if let Some(latch) = job_map.latch_of(query) {
        for (i, waiter) in latch.info.lock().waiters.iter().enumerate() {
            if let Some(waiter_query) = waiter.query {
                // Return a value which indicates that this waiter can be resumed
                visit(waiter.span, waiter_query).map_break(|_| Some((query, i)))?;
            }
        }
    }

    ControlFlow::Continue(())
}

/// Look for query cycles by doing a depth first search starting at `query`.
/// `span` is the reason for the `query` to execute. This is initially DUMMY_SP.
/// If a cycle is detected, this initial value is replaced with the span causing
/// the cycle.
fn cycle_check<'tcx>(
    job_map: &QueryJobMap<'tcx>,
    query: QueryJobId,
    span: Span,
    stack: &mut Vec<(Span, QueryJobId)>,
    visited: &mut FxHashSet<QueryJobId>,
) -> ControlFlow<Option<Waiter>> {
    if !visited.insert(query) {
        return if let Some(p) = stack.iter().position(|q| q.1 == query) {
            // We detected a query cycle, fix up the initial span and return Some

            // Remove previous stack entries
            stack.drain(0..p);
            // Replace the span for the first query with the cycle cause
            stack[0].0 = span;
            ControlFlow::Break(None)
        } else {
            ControlFlow::Continue(())
        };
    }

    // Query marked as visited is added it to the stack
    stack.push((span, query));

    // Visit all the waiters
    let r = visit_waiters(job_map, query, |span, successor| {
        cycle_check(job_map, successor, span, stack, visited)
    });

    // Remove the entry in our stack if we didn't find a cycle
    if r.is_continue() {
        stack.pop();
    }

    r
}

/// Finds out if there's a path to the compiler root (aka. code which isn't in a query)
/// from `query` without going through any of the queries in `visited`.
/// This is achieved with a depth first search.
fn connected_to_root<'tcx>(
    job_map: &QueryJobMap<'tcx>,
    query: QueryJobId,
    visited: &mut FxHashSet<QueryJobId>,
) -> ControlFlow<Option<Waiter>> {
    // We already visited this or we're deliberately ignoring it
    if !visited.insert(query) {
        return ControlFlow::Continue(());
    }

    // This query is connected to the root (it has no query parent), return true
    if job_map.parent_of(query).is_none() {
        return ControlFlow::Break(None);
    }

    visit_waiters(job_map, query, |_, successor| connected_to_root(job_map, successor, visited))
}

// Deterministically pick an query from a list
fn pick_query<'a, 'tcx, T, F>(job_map: &QueryJobMap<'tcx>, queries: &'a [T], f: F) -> &'a T
where
    F: Fn(&T) -> (Span, QueryJobId),
{
    // Deterministically pick an entry point
    // FIXME: Sort this instead
    queries
        .iter()
        .min_by_key(|v| {
            let (span, query) = f(v);
            let hash = job_map.frame_of(query).hash;
            // Prefer entry points which have valid spans for nicer error messages
            // We add an integer to the tuple ensuring that entry points
            // with valid spans are picked first
            let span_cmp = if span == DUMMY_SP { 1 } else { 0 };
            (span_cmp, hash)
        })
        .unwrap()
}

/// Looks for query cycles starting from the last query in `jobs`.
/// If a cycle is found, all queries in the cycle is removed from `jobs` and
/// the function return true.
/// If a cycle was not found, the starting query is removed from `jobs` and
/// the function returns false.
fn remove_cycle<'tcx>(
    job_map: &QueryJobMap<'tcx>,
    jobs: &mut Vec<QueryJobId>,
    wakelist: &mut Vec<Arc<QueryWaiter<'tcx>>>,
) -> bool {
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();
    // Look for a cycle starting with the last query in `jobs`
    if let ControlFlow::Break(waiter) =
        cycle_check(job_map, jobs.pop().unwrap(), DUMMY_SP, &mut stack, &mut visited)
    {
        // The stack is a vector of pairs of spans and queries; reverse it so that
        // the earlier entries require later entries
        let (mut spans, queries): (Vec<_>, Vec<_>) = stack.into_iter().rev().unzip();

        // Shift the spans so that queries are matched with the span for their waitee
        spans.rotate_right(1);

        // Zip them back together
        let mut stack: Vec<_> = iter::zip(spans, queries).collect();

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
                if job_map.parent_of(query).is_none() {
                    // This query is connected to the root (it has no query parent)
                    Some((span, query, None))
                } else {
                    let mut waiters = Vec::new();
                    // Find all the direct waiters who lead to the root
                    let _ = visit_waiters(job_map, query, |span, waiter| {
                        // Mark all the other queries in the cycle as already visited
                        let mut visited = FxHashSet::from_iter(stack.iter().map(|q| q.1));

                        if connected_to_root(job_map, waiter, &mut visited).is_break() {
                            waiters.push((span, waiter));
                        }

                        ControlFlow::Continue(())
                    });
                    if waiters.is_empty() {
                        None
                    } else {
                        // Deterministically pick one of the waiters to show to the user
                        let waiter = *pick_query(job_map, &waiters, |s| *s);
                        Some((span, query, Some(waiter)))
                    }
                }
            })
            .collect::<Vec<(Span, QueryJobId, Option<(Span, QueryJobId)>)>>();

        // Deterministically pick an entry point
        let (_, entry_point, usage) = pick_query(job_map, &entry_points, |e| (e.0, e.1));

        // Shift the stack so that our entry point is first
        let entry_point_pos = stack.iter().position(|(_, query)| query == entry_point);
        if let Some(pos) = entry_point_pos {
            stack.rotate_left(pos);
        }

        let usage = usage.map(|(span, job)| (span, job_map.frame_of(job).clone()));

        // Create the cycle error
        let error = CycleError {
            usage,
            cycle: stack
                .iter()
                .map(|&(span, job)| QueryInfo { span, frame: job_map.frame_of(job).clone() })
                .collect(),
        };

        // We unwrap `waiter` here since there must always be one
        // edge which is resumable / waited using a query latch
        let (waitee_query, waiter_idx) = waiter.unwrap();

        // Extract the waiter we want to resume
        let waiter = job_map.latch_of(waitee_query).unwrap().extract_waiter(waiter_idx);

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
pub fn break_query_cycles<'tcx>(
    job_map: QueryJobMap<'tcx>,
    registry: &rustc_thread_pool::Registry,
) {
    let mut wakelist = Vec::new();
    // It is OK per the comments:
    // - https://github.com/rust-lang/rust/pull/131200#issuecomment-2798854932
    // - https://github.com/rust-lang/rust/pull/131200#issuecomment-2798866392
    #[allow(rustc::potential_query_instability)]
    let mut jobs: Vec<QueryJobId> = job_map.map.keys().copied().collect();

    let mut found_cycle = false;

    while jobs.len() > 0 {
        if remove_cycle(&job_map, &mut jobs, &mut wakelist) {
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
    if !found_cycle {
        panic!(
            "deadlock detected as we're unable to find a query cycle to break\n\
            current query map:\n{job_map:#?}",
        );
    }

    // Mark all the thread we're about to wake up as unblocked. This needs to be done before
    // we wake the threads up as otherwise Rayon could detect a deadlock if a thread we
    // resumed fell asleep and this thread had yet to mark the remaining threads as unblocked.
    for _ in 0..wakelist.len() {
        rustc_thread_pool::mark_unblocked(registry);
    }

    for waiter in wakelist.into_iter() {
        waiter.condvar.notify_one();
    }
}

pub fn print_query_stack<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut current_query: Option<QueryJobId>,
    dcx: DiagCtxtHandle<'_>,
    limit_frames: Option<usize>,
    mut file: Option<std::fs::File>,
) -> usize {
    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `DiagCtxt` may be in a weird
    // state if it was responsible for triggering the panic.
    let mut count_printed = 0;
    let mut count_total = 0;

    // Make use of a partial query job map if we fail to take locks collecting active queries.
    let job_map: QueryJobMap<'_> = collect_active_jobs_from_all_queries(tcx, false)
        .unwrap_or_else(|partial_job_map| partial_job_map);

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let Some(query_info) = job_map.map.get(&query) else {
            break;
        };
        let query_extra = query_info.frame.info.extract();
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            dcx.struct_failure_note(format!(
                "#{} [{:?}] {}",
                count_printed, query_info.frame.dep_kind, query_extra.description
            ))
            .with_span(query_info.job.span)
            .emit();
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{} [{:?}] {}",
                count_total, query_info.frame.dep_kind, query_extra.description
            );
        }

        current_query = query_info.job.parent;
        count_total += 1;
    }

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "end of query stack");
    }
    count_total
}

#[inline(never)]
#[cold]
pub(crate) fn report_cycle<'a>(
    sess: &'a Session,
    CycleError { usage, cycle: stack }: &CycleError,
) -> Diag<'a> {
    assert!(!stack.is_empty());

    let span = stack[0].frame.info.default_span(stack[1 % stack.len()].span);

    let mut cycle_stack = Vec::new();

    use crate::error::StackCount;
    let stack_count = if stack.len() == 1 { StackCount::Single } else { StackCount::Multiple };

    for i in 1..stack.len() {
        let frame = &stack[i].frame;
        let span = frame.info.default_span(stack[(i + 1) % stack.len()].span);
        cycle_stack
            .push(crate::error::CycleStack { span, desc: frame.info.description.to_owned() });
    }

    let mut cycle_usage = None;
    if let Some((span, ref query)) = *usage {
        cycle_usage = Some(crate::error::CycleUsage {
            span: query.info.default_span(span),
            usage: query.info.description.to_string(),
        });
    }

    let alias =
        if stack.iter().all(|entry| matches!(entry.frame.info.def_kind, Some(DefKind::TyAlias))) {
            Some(crate::error::Alias::Ty)
        } else if stack.iter().all(|entry| entry.frame.info.def_kind == Some(DefKind::TraitAlias)) {
            Some(crate::error::Alias::Trait)
        } else {
            None
        };

    let cycle_diag = crate::error::Cycle {
        span,
        cycle_stack,
        stack_bottom: stack[0].frame.info.description.to_owned(),
        alias,
        cycle_usage,
        stack_count,
        note_span: (),
    };

    sess.dcx().create_err(cycle_diag)
}
