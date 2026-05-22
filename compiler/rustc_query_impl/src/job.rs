use std::io::Write;
use std::ops::ControlFlow;
use std::{iter, mem, ptr};

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_middle::queries::TaggedQueryKey;
use rustc_middle::query::{Cycle, QueryJob, QueryJobRef, QueryStackFrame, QueryWaiterGuard};
use rustc_middle::ty::TyCtxt;
use rustc_span::{DUMMY_SP, Span};

pub(crate) fn _find_cycle_in_stack<'a, 'tcx>(
    top_job: QueryJobRef<'a, 'tcx>,
    mut current_job: Option<QueryJobRef<'a, 'tcx>>,
    span: Span,
) -> Cycle<'tcx> {
    // Find the waitee amongst `current_job` parents.
    let mut frames = Vec::new();

    while let Some(job) = current_job {
        frames.push(QueryStackFrame { span: job.span, tagged_key: (job.form_tagged_key)() });

        if ptr::eq(job, top_job) {
            frames.reverse();

            // This is the end of the cycle. The span entry we included was for
            // the usage of the cycle itself, and not part of the cycle.
            // Replace it with the span which caused the cycle to form.
            frames[0].span = span;
            // Find out why the cycle itself was used.
            let usage = try {
                let parent = job.parent?;
                QueryStackFrame { span: job.span, tagged_key: (parent.form_tagged_key)() }
            };
            return Cycle { usage, frames };
        }

        current_job = job.parent;
    }

    panic!("did not find a cycle")
}

/// Finds the query job closest to the root that is for the same query method as `id`
/// (but not necessarily the same query key), and returns information about it.
#[cold]
#[inline(never)]
pub(crate) fn find_dep_kind_root<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    mut job: QueryJobRef<'a, 'tcx>,
) -> (Span, String, usize) {
    let mut depth = 1;
    // Two query jobs are for the same query method if they have the same
    // `TaggedQueryKey` discriminant.
    let mut tagged_key = (job.form_tagged_key)();
    let expected_query = mem::discriminant::<TaggedQueryKey<'tcx>>(&tagged_key);

    while let Some(next_job) = job.parent {
        let next_tagged_key = (next_job.form_tagged_key)();
        if mem::discriminant(&next_tagged_key) == expected_query {
            depth += 1;
            job = next_job;
            tagged_key = next_tagged_key;
        }
    }
    (job.span, tagged_key.description(tcx), depth)
}

/// The locaton of a resumable waiter. The usize is the thread index into worker thread pool.
// FIXME: correct this comment:
/// We'll use this to remove the waiter using `QueryLatch::extract_waiter` if we're waking it up.
type ResumableWaiterLocation<'a, 'tcx> = (QueryJobRef<'a, 'tcx>, usize);

/// This abstracts over non-resumable waiters which are found in `QueryJob`'s `parent` field
/// and resumable waiters are in `latch` field.
struct AbstractedWaiter<'a, 'tcx> {
    /// The span corresponding to the reason for why we're waiting on this query.
    span: Span,
    /// The query which we are waiting from, if none the waiter is from a compiler root.
    parent: Option<QueryJobRef<'a, 'tcx>>,
    resumable: Option<ResumableWaiterLocation<'a, 'tcx>>,
}

/// Returns all the non-resumable and resumable waiters of a query.
/// This is used so we can uniformly loop over both non-resumable and resumable waiters.
fn abstracted_waiters_of<'a, 'tcx>(
    query: QueryJobRef<'a, 'tcx>,
    waiters: &'a [Option<QueryWaiterGuard<'a, 'tcx>>],
) -> Vec<AbstractedWaiter<'a, 'tcx>> {
    let mut result = Vec::new();

    // Add the parent which is a non-resumable waiter since it's on the same stack
    result.push(AbstractedWaiter { span: query.span, parent: query.parent, resumable: None });

    // Add the explicit waiters which use condvars and are resumable
    let worker_threads = query.entry_status.waiter_threads();
    if worker_threads != 0 {
        for i in 0..rustc_thread_pool::max_num_threads() {
            if worker_threads & (1 << i) != 0 {
                let waiter = waiters[i as usize].as_ref().unwrap();
                result.push(AbstractedWaiter {
                    span: waiter.span(),
                    parent: waiter.parent(),
                    resumable: Some((query, i as usize)),
                });
            }
        }
    }

    result
}

/// Looks for a query cycle by doing a depth first search starting at `query`.
/// `span` is the reason for the `query` to execute. This is initially DUMMY_SP.
/// If a cycle is detected, this initial value is replaced with the span causing
/// the cycle. `stack` will contain just the cycle on return if detected.
fn find_cycle<'a, 'tcx>(
    query: QueryJobRef<'a, 'tcx>,
    span: Span,
    waiters: &'a [Option<QueryWaiterGuard<'a, 'tcx>>],
    stack: &mut Vec<(Span, QueryJobRef<'a, 'tcx>)>,
    visited: &mut FxHashSet<*const QueryJob<'a, 'tcx>>,
) -> ControlFlow<Option<ResumableWaiterLocation<'a, 'tcx>>> {
    if !visited.insert(query) {
        return if let Some(pos) = stack.iter().position(|q| ptr::eq(q.1, query)) {
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
    for abstracted_waiter in abstracted_waiters_of(query, waiters) {
        let Some(parent) = abstracted_waiter.parent else {
            // Skip waiters which are not queries
            continue;
        };
        if let ControlFlow::Break(maybe_resumable) =
            find_cycle(parent, abstracted_waiter.span, waiters, stack, visited)
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
fn connected_to_root<'a, 'tcx>(
    query: QueryJobRef<'a, 'tcx>,
    waiters: &'a [Option<QueryWaiterGuard<'a, 'tcx>>],
    visited: &mut FxHashSet<*const QueryJob<'a, 'tcx>>,
) -> bool {
    // We already visited this or we're deliberately ignoring it
    if !visited.insert(query) {
        return false;
    }

    // Visit all the waiters
    for abstracted_waiter in abstracted_waiters_of(query, waiters) {
        match abstracted_waiter.parent {
            // This query is connected to the root
            None => return true,
            Some(parent) => {
                if connected_to_root(parent, waiters, visited) {
                    return true;
                }
            }
        }
    }

    false
}

/// Processes a found query cycle into a `Cycle`
fn process_cycle<'a, 'tcx>(
    stack: Vec<(Span, QueryJobRef<'a, 'tcx>)>,
    waiters: &'a [Option<QueryWaiterGuard<'a, 'tcx>>],
) -> Cycle<'tcx> {
    // The stack is a vector of pairs of spans and queries; reverse it so that
    // the earlier entries require later entries
    let (mut spans, queries): (Vec<_>, Vec<_>) = stack.into_iter().rev().unzip();

    // Shift the spans so that queries are matched with the span for their waitee
    spans.rotate_right(1);

    // Zip them back together
    let mut stack: Vec<_> = iter::zip(spans, queries).collect();

    struct EntryPoint<'a, 'tcx> {
        query_in_cycle: QueryJobRef<'a, 'tcx>,
        query_waiting_on_cycle: Option<(Span, QueryJobRef<'a, 'tcx>)>,
    }

    // Find the queries in the cycle which are
    // connected to queries outside the cycle
    let entry_points = stack
        .iter()
        .filter_map(|&(_, query_in_cycle)| {
            let mut entrypoint = false;
            let mut query_waiting_on_cycle = None;

            // Find a direct waiter who leads to the root
            for abstracted_waiter in abstracted_waiters_of(query_in_cycle, waiters) {
                let Some(parent) = abstracted_waiter.parent else {
                    // The query in the cycle is directly connected to root.
                    entrypoint = true;
                    continue;
                };

                // Mark all the other queries in the cycle as already visited,
                // so paths to the root through the cycle itself won't count.
                let mut visited = FxHashSet::from_iter(stack.iter().map(|q| q.1 as *const _));

                if connected_to_root(parent, waiters, &mut visited) {
                    query_waiting_on_cycle = Some((abstracted_waiter.span, parent));
                    entrypoint = true;
                    break;
                }
            }

            entrypoint.then_some(EntryPoint { query_in_cycle, query_waiting_on_cycle })
        })
        .collect::<Vec<EntryPoint<'a, 'tcx>>>();

    // Pick an entry point, preferring ones with waiters
    let entry_point = entry_points
        .iter()
        .find(|entry_point| entry_point.query_waiting_on_cycle.is_some())
        .unwrap_or(&entry_points[0]);

    // Shift the stack so that our entry point is first
    let entry_point_pos =
        stack.iter().position(|(_, query)| ptr::eq(*query, entry_point.query_in_cycle));
    if let Some(pos) = entry_point_pos {
        stack.rotate_left(pos);
    }

    let usage = entry_point
        .query_waiting_on_cycle
        .map(|(span, job)| QueryStackFrame { span, tagged_key: (job.form_tagged_key)() });

    // Create the cycle error
    Cycle {
        usage,
        frames: stack
            .iter()
            .map(|&(span, job)| QueryStackFrame { span, tagged_key: (job.form_tagged_key)() })
            .collect(),
    }
}

/// Looks for a query cycle starting at `query`.
/// Returns a waiter to resume if a cycle is found.
fn find_and_process_cycle<'a, 'tcx>(
    query: QueryJobRef<'a, 'tcx>,
    waiters: &'a [Option<QueryWaiterGuard<'a, 'tcx>>],
) -> Option<(usize, Cycle<'tcx>)> {
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();
    if let ControlFlow::Break(resumable) =
        find_cycle(query, DUMMY_SP, waiters, &mut stack, &mut visited)
    {
        // Create the cycle error
        let error = process_cycle(stack, waiters);

        // We unwrap `resumable` here since there must always be one
        // edge which is resumable / waited using a query latch
        let (waitee_query, thread_idx) = resumable.unwrap();

        // Remove a query waiter we want to resume
        waitee_query.entry_status.remove_waiter_threads(1 << thread_idx);

        // Put the waiter on the list of things to resume
        Some((thread_idx, error))
    } else {
        None
    }
}

/// Detects query cycles by using depth first search over all active query jobs.
/// If a query cycle is found it will break the cycle by finding an edge which
/// uses a query latch and then resuming that waiter.
///
/// There may be multiple cycles involved in a deadlock, but this only breaks one at a time so
/// there will be multiple rounds through the deadlock handler if multiple cycles are present.
#[allow(rustc::potential_query_instability)]
pub fn break_query_cycle<'tcx>(tcx: TyCtxt<'tcx>, registry: &rustc_thread_pool::Registry) {
    let mut waiters: Box<[Option<QueryWaiterGuard<'_, 'tcx>>]> = (0..registry.num_threads())
        .map(|thrd_idx| tcx.parking_area.lock_waiter(thrd_idx))
        .collect();

    let mut iter = waiters.iter().flatten();
    // Look for a cycle starting at each query job
    let (waiter_idx, cycle) = 'cycle: loop {
        let waiter = iter.next().expect("unable to find a query cycle");
        let mut parent = waiter.parent();
        while let Some(query) = parent {
            if let Some(res) = find_and_process_cycle(query, &waiters) {
                break 'cycle res;
            }
            parent = query.parent;
        }
    };

    // Mark the thread we're about to wake up as unblocked.
    waiters[waiter_idx].take().unwrap().unpark_with_cycle(cycle, registry);
}

pub fn print_query_stack<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    mut current_query: Option<QueryJobRef<'a, 'tcx>>,
    dcx: DiagCtxtHandle<'_>,
    limit_frames: Option<usize>,
    mut file: Option<std::fs::File>,
) -> usize {
    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `DiagCtxt` may be in a weird
    // state if it was responsible for triggering the panic.
    let mut count_printed = 0;
    let mut count_total = 0;

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let tagged_key = (query.form_tagged_key)();
        let description = tagged_key.description(tcx);
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            dcx.struct_failure_note(format!(
                "#{count_printed} [{query_name}] {description}",
                query_name = tagged_key.query_name(),
            ))
            .with_span(query.span)
            .emit();
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{count_total} [{query_name}] {description}",
                query_name = tagged_key.query_name(),
            );
        }

        current_query = query.parent;
        count_total += 1;
    }

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "end of query stack");
    }
    count_total
}

#[inline(never)]
#[cold]
pub(crate) fn create_cycle_error<'tcx>(
    tcx: TyCtxt<'tcx>,
    Cycle { usage, frames }: &Cycle<'tcx>,
    nested: bool,
) -> Diag<'tcx> {
    assert!(!frames.is_empty());

    let span = frames[0].tagged_key.catch_default_span(tcx, frames[1 % frames.len()].span);

    let mut cycle_stack = Vec::new();

    use crate::error::StackCount;
    let stack_bottom = frames[0].tagged_key.catch_description(tcx);
    let stack_count = if frames.len() == 1 {
        StackCount::Single { stack_bottom: stack_bottom.clone() }
    } else {
        StackCount::Multiple { stack_bottom: stack_bottom.clone() }
    };

    for i in 1..frames.len() {
        let frame = &frames[i];
        let span = frame.tagged_key.catch_default_span(tcx, frames[(i + 1) % frames.len()].span);
        cycle_stack
            .push(crate::error::CycleStack { span, desc: frame.tagged_key.catch_description(tcx) });
    }

    let cycle_usage = usage.as_ref().map(|usage| crate::error::CycleUsage {
        span: usage.tagged_key.catch_default_span(tcx, usage.span),
        usage: usage.tagged_key.catch_description(tcx),
    });

    let is_all_def_kind = |def_kind| {
        // Trivial type alias and trait alias cycles consists of `type_of` and
        // `explicit_implied_predicates_of` queries, so we just check just these here.
        frames.iter().all(|frame| match frame.tagged_key {
            TaggedQueryKey::type_of(def_id)
            | TaggedQueryKey::explicit_implied_predicates_of(def_id)
                if tcx.def_kind(def_id) == def_kind =>
            {
                true
            }
            _ => false,
        })
    };

    let alias = if !nested {
        if is_all_def_kind(DefKind::TyAlias) {
            Some(crate::error::Alias::Ty)
        } else if is_all_def_kind(DefKind::TraitAlias) {
            Some(crate::error::Alias::Trait)
        } else {
            None
        }
    } else {
        None
    };

    if nested {
        tcx.sess.dcx().create_err(crate::error::NestedCycle {
            span,
            cycle_stack,
            stack_bottom: crate::error::NestedCycleBottom { stack_bottom },
            cycle_usage,
            stack_count,
            note_span: (),
        })
    } else {
        tcx.sess.dcx().create_err(crate::error::Cycle {
            span,
            cycle_stack,
            stack_bottom,
            alias,
            cycle_usage,
            stack_count,
            note_span: (),
        })
    }
}
