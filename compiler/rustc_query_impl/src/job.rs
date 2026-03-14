use std::io::Write;
use std::iter;
use std::ops::ControlFlow;
use std::sync::Arc;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_middle::query::{
    CycleError, QueryInfo, QueryJob, QueryJobId, QueryLatch, QueryStackFrame, QueryWaiter,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::{DUMMY_SP, Span};

use crate::{CollectActiveJobsKind, collect_active_jobs_from_all_queries};

/// Map from query job IDs to job information collected by
/// `collect_active_jobs_from_all_queries`.
#[derive(Debug, Default)]
pub struct QueryJobMap<'tcx> {
    map: FxHashMap<QueryJobId, QueryJobInfo<'tcx>>,
}

impl<'tcx> QueryJobMap<'tcx> {
    /// Adds information about a job ID to the job map.
    ///
    /// Should only be called by `gather_active_jobs`.
    pub(crate) fn insert(&mut self, id: QueryJobId, info: QueryJobInfo<'tcx>) {
        self.map.insert(id, info);
    }

    fn frame_of(&self, id: QueryJobId) -> &QueryStackFrame<'tcx> {
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
    pub(crate) frame: QueryStackFrame<'tcx>,
    pub(crate) job: QueryJob<'tcx>,
}

pub(crate) fn find_cycle_in_stack<'tcx>(
    id: QueryJobId,
    job_map: QueryJobMap<'tcx>,
    current_job: &Option<QueryJobId>,
    span: Span,
) -> CycleError<'tcx> {
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

/// The locaton of a resumable waiter. The usize is the index into waiters in the query's latch.
/// We'll use this to remove the waiter using `QueryLatch::extract_waiter` if we're waking it up.
type ResumableWaiterLocation = (QueryJobId, usize);

/// This abstracts over non-resumable waiters which are found in `QueryJob`'s `parent` field
/// and resumable waiters are in `latch` field.
struct AbstractedWaiter {
    /// The span corresponding to the reason for why we're waiting on this query.
    span: Span,
    /// The query which we are waiting from, if none the waiter is from a compiler root.
    parent: Option<QueryJobId>,
    resumable: Option<ResumableWaiterLocation>,
}

/// Returns all the non-resumable and resumable waiters of a query.
/// This is used so we can uniformly loop over both non-resumable and resumable waiters.
fn abstracted_waiters_of(job_map: &QueryJobMap<'_>, query: QueryJobId) -> Vec<AbstractedWaiter> {
    let mut result = Vec::new();

    // Add the parent which is a non-resumable waiter since it's on the same stack
    result.push(AbstractedWaiter {
        span: job_map.span_of(query),
        parent: job_map.parent_of(query),
        resumable: None,
    });

    // Add the explicit waiters which use condvars and are resumable
    if let Some(latch) = job_map.latch_of(query) {
        for (i, waiter) in latch.waiters.lock().as_ref().unwrap().iter().enumerate() {
            result.push(AbstractedWaiter {
                span: waiter.span,
                parent: waiter.parent,
                resumable: Some((query, i)),
            });
        }
    }

    result
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
) -> ControlFlow<Option<ResumableWaiterLocation>> {
    if !visited.insert(query) {
        return if let Some(pos) = stack.iter().position(|q| q.1 == query) {
            // We detected a query cycle, fix up the initial span and return Some

            // Remove previous stack entries
            stack.drain(0..pos);
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
    for abstracted_waiter in abstracted_waiters_of(job_map, query) {
        let Some(parent) = abstracted_waiter.parent else {
            // Skip waiters which are not queries
            continue;
        };
        if let ControlFlow::Break(maybe_resumable) =
            cycle_check(job_map, parent, abstracted_waiter.span, stack, visited)
        {
            // Return the resumable waiter in `waiter.resumable` if present
            return ControlFlow::Break(abstracted_waiter.resumable.or(maybe_resumable));
        }
    }

    // Remove the entry in our stack since we didn't find a cycle
    stack.pop();

    ControlFlow::Continue(())
}

/// Finds out if there's a path to the compiler root (aka. code which isn't in a query)
/// from `query` without going through any of the queries in `visited`.
/// This is achieved with a depth first search.
fn connected_to_root<'tcx>(
    job_map: &QueryJobMap<'tcx>,
    query: QueryJobId,
    visited: &mut FxHashSet<QueryJobId>,
) -> bool {
    // We already visited this or we're deliberately ignoring it
    if !visited.insert(query) {
        return false;
    }

    // Visit all the waiters
    for abstracted_waiter in abstracted_waiters_of(job_map, query) {
        match abstracted_waiter.parent {
            // This query is connected to the root
            None => return true,
            Some(parent) => {
                if connected_to_root(job_map, parent, visited) {
                    return true;
                }
            }
        }
    }

    false
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
    if let ControlFlow::Break(resumable) =
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

        struct EntryPoint {
            query_in_cycle: QueryJobId,
            query_waiting_on_cycle: Option<(Span, QueryJobId)>,
        }

        // Find the queries in the cycle which are
        // connected to queries outside the cycle
        let entry_points = stack
            .iter()
            .filter_map(|&(_, query_in_cycle)| {
                let mut entrypoint = false;
                let mut query_waiting_on_cycle = None;

                // Find a direct waiter who leads to the root
                for abstracted_waiter in abstracted_waiters_of(job_map, query_in_cycle) {
                    let Some(parent) = abstracted_waiter.parent else {
                        // The query in the cycle is directly connected to root.
                        entrypoint = true;
                        continue;
                    };

                    // Mark all the other queries in the cycle as already visited,
                    // so paths to the root through the cycle itself won't count.
                    let mut visited = FxHashSet::from_iter(stack.iter().map(|q| q.1));

                    if connected_to_root(job_map, parent, &mut visited) {
                        query_waiting_on_cycle = Some((abstracted_waiter.span, parent));
                        entrypoint = true;
                        break;
                    }
                }

                entrypoint.then_some(EntryPoint { query_in_cycle, query_waiting_on_cycle })
            })
            .collect::<Vec<EntryPoint>>();

        // Pick an entry point, preferring ones with waiters
        let entry_point = entry_points
            .iter()
            .find(|entry_point| entry_point.query_waiting_on_cycle.is_some())
            .unwrap_or(&entry_points[0]);

        // Shift the stack so that our entry point is first
        let entry_point_pos =
            stack.iter().position(|(_, query)| *query == entry_point.query_in_cycle);
        if let Some(pos) = entry_point_pos {
            stack.rotate_left(pos);
        }

        let usage = entry_point
            .query_waiting_on_cycle
            .map(|(span, job)| (span, job_map.frame_of(job).clone()));

        // Create the cycle error
        let error = CycleError {
            usage,
            cycle: stack
                .iter()
                .map(|&(span, job)| QueryInfo { span, frame: job_map.frame_of(job).clone() })
                .collect(),
        };

        // We unwrap `resumable` here since there must always be one
        // edge which is resumable / waited using a query latch
        let (waitee_query, waiter_idx) = resumable.unwrap();

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
    let job_map = collect_active_jobs_from_all_queries(tcx, CollectActiveJobsKind::PartialAllowed);

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let Some(query_info) = job_map.map.get(&query) else {
            break;
        };
        let description = query_info.frame.tagged_key.description(tcx);
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            dcx.struct_failure_note(format!(
                "#{} [{:?}] {}",
                count_printed, query_info.frame.dep_kind, description
            ))
            .with_span(query_info.job.span)
            .emit();
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{} [{:?}] {}",
                count_total, query_info.frame.dep_kind, description
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
pub(crate) fn report_cycle<'tcx>(
    tcx: TyCtxt<'tcx>,
    CycleError { usage, cycle: stack }: &CycleError<'tcx>,
) -> Diag<'tcx> {
    assert!(!stack.is_empty());

    let span = stack[0].frame.tagged_key.default_span(tcx, stack[1 % stack.len()].span);

    let mut cycle_stack = Vec::new();

    use crate::error::StackCount;
    let stack_bottom = stack[0].frame.tagged_key.description(tcx);
    let stack_count = if stack.len() == 1 {
        StackCount::Single { stack_bottom: stack_bottom.clone() }
    } else {
        StackCount::Multiple { stack_bottom: stack_bottom.clone() }
    };

    for i in 1..stack.len() {
        let frame = &stack[i].frame;
        let span = frame.tagged_key.default_span(tcx, stack[(i + 1) % stack.len()].span);
        cycle_stack
            .push(crate::error::CycleStack { span, desc: frame.tagged_key.description(tcx) });
    }

    let mut cycle_usage = None;
    if let Some((span, ref query)) = *usage {
        cycle_usage = Some(crate::error::CycleUsage {
            span: query.tagged_key.default_span(tcx, span),
            usage: query.tagged_key.description(tcx),
        });
    }

    let alias = if stack
        .iter()
        .all(|entry| matches!(entry.frame.tagged_key.def_kind(tcx), Some(DefKind::TyAlias)))
    {
        Some(crate::error::Alias::Ty)
    } else if stack
        .iter()
        .all(|entry| entry.frame.tagged_key.def_kind(tcx) == Some(DefKind::TraitAlias))
    {
        Some(crate::error::Alias::Trait)
    } else {
        None
    };

    let cycle_diag = crate::error::Cycle {
        span,
        cycle_stack,
        stack_bottom,
        alias,
        cycle_usage,
        stack_count,
        note_span: (),
    };

    tcx.sess.dcx().create_err(cycle_diag)
}
