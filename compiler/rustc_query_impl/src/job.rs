use std::collections::BTreeMap;
use std::io::Write;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexmap::{self, IndexMap};
use rustc_data_structures::tree_node_index::TreeNodeIndex;
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_middle::query::{
    CycleError, QueryInfo, QueryJob, QueryJobId, QueryStackDeferred, QueryStackFrame,
};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::Span;

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
}

#[derive(Clone, Debug)]
pub(crate) struct QueryJobInfo<'tcx> {
    pub(crate) frame: QueryStackFrame<QueryStackDeferred<'tcx>>,
    pub(crate) job: QueryJob<'tcx>,
}

pub(crate) fn find_cycle_in_stack<'tcx>(
    id: QueryJobId,
    job_map: QueryJobMap<'tcx>,
    mut current_job: Option<QueryJobId>,
    span: Span,
) -> CycleError<QueryStackDeferred<'tcx>> {
    // Find the waitee amongst `current_job` parents
    let mut cycle = Vec::new();

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
                (info.job.span, job_map.frame_of(parent.id).clone())
            };
            return CycleError { usage, cycle };
        }

        current_job = info.job.parent.map(|i| i.id);
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
    let mut current = info.job.parent;
    let mut last_layout = (info.clone(), depth);

    while let Some(inclusion) = current {
        let info = &job_map.map[&inclusion.id];
        if info.frame.dep_kind == dep_kind {
            depth += 1;
            last_layout = (info.clone(), depth);
        }
        current = info.job.parent;
    }
    last_layout
}

/// Breaks left-most cycle on a left-most query in order of a single-threaded execution.
///
/// Order of queries is tracked using [`TreeNodeIndex`] in [`rustc_middle::sync`].
/// This function uses ordered depth-first search from a single root query down to the first
/// duplicate query.
/// It doesn't distinguish between a query wait and a query execution, so both are just query calls.
/// As such some queries may have two or more parent query calls too.
///
/// But while it breaks on the same query as with a single thread,
/// we are not guaranteed to break on the same query **call**.
/// This is good enough, as the difference is irrelevant to query cycle recovery code.
/// Every other difference AFAIK is tolerable.
/// Potential different query result values are fine as either ill-defined due to cycles or
/// as they preserve the same query result value between different query calls.
///
/// To illustrate how it work say we have a query cycle:
///
/// ```text
/// a() -> b() -> a()
/// ```
///
/// and a program `join(|| a(), || b())`.
/// On a single-thread it triggers cycle recovery on a `a()` call within `b()` query.
/// However consider a multi-threaded execution:
///
/// ```text
/// thread 1: waits on a()
/// thread 2: b() -> a() -> waits on b()
/// ```
///
/// Similar to single-threaded execution, we have to resume wait on `a()`.
/// However this time it could only be done to a *different query call*, the one inside of `join`.
/// Then we resume until thread 1 blocks in join on a `b()` task.
///
/// ```text
/// thread 1: indirectly waits on b()
/// thread 2: b() -> a() -> waits on b()
/// ```
///
/// Now the left-most query to break is `b()` so we resume thread 2.
/// This difference in behavior is strictly more tolerable than the undeterministic cycle breaking.
#[allow(rustc::potential_query_instability)]
pub fn break_query_cycles<'tcx>(
    query_map: QueryJobMap<'tcx>,
    registry: &rustc_thread_pool::Registry,
) {
    let mut root_query = None;
    for (&query, info) in &query_map.map {
        if info.job.parent.is_none() {
            assert!(root_query.is_none(), "found multiple threads without start");
            root_query = Some(query);
        }
    }
    let root_query = root_query.expect("no root query was found");

    let mut subqueries = FxHashMap::<_, BTreeMap<TreeNodeIndex, _>>::default();
    for query in query_map.map.values() {
        let Some(inclusion) = &query.job.parent else {
            continue;
        };
        let old = subqueries
            .entry(inclusion.id)
            .or_default()
            .insert(inclusion.branch, (query.job.id, usize::MAX));
        assert!(old.is_none());
    }

    for query in query_map.map.values() {
        let Some(latch) = &query.job.latch else {
            continue;
        };
        // Latch mutexes should be at least about to unlock as we do not hold it anywhere too long
        let lock = latch.info.lock();
        assert!(!lock.complete);
        for (waiter_idx, waiter) in lock.waiters.iter().enumerate() {
            let inclusion = waiter.query.expect("cannot wait on a root query");
            let old = subqueries
                .entry(inclusion.id)
                .or_default()
                .insert(inclusion.branch, (query.job.id, waiter_idx));
            assert!(old.is_none());
        }
    }

    let mut visited = IndexMap::new();
    let mut last_usage = None;
    let mut last_waiter_idx = usize::MAX;
    let mut current = root_query;
    while let indexmap::map::Entry::Vacant(entry) = visited.entry(current) {
        entry.insert((last_usage, last_waiter_idx));
        last_usage = Some(current);
        (current, last_waiter_idx) = *subqueries
            .get(&current)
            .unwrap_or_else(|| {
                panic!(
                    "deadlock detected as we're unable to find a query cycle to break\n\
                current query map:\n{:#?}",
                    query_map
                )
            })
            .first_key_value()
            .unwrap()
            .1;
    }
    let usage = visited[&current].0;
    let mut iter = visited.keys().rev();
    let mut cycle = Vec::new();
    loop {
        let query_id = *iter.next().unwrap();
        let query = &query_map.map[&query_id];
        cycle.push(QueryInfo { span: query.job.span, frame: query.frame.clone() });
        if query_id == current {
            break;
        }
    }

    cycle.reverse();
    let cycle_error = CycleError {
        usage: usage.map(|id| {
            let query = &query_map.map[&id];
            (query.job.span, query.frame.clone())
        }),
        cycle,
    };

    let (waited_on, waiter_idx) = if last_waiter_idx != usize::MAX {
        (current, last_waiter_idx)
    } else {
        let (&waited_on, &(_, waiter_idx)) =
            visited.iter().rev().find(|(_, (_, waiter_idx))| *waiter_idx != usize::MAX).unwrap();
        (waited_on, waiter_idx)
    };
    let waited_on = &query_map.map[&waited_on];
    let latch = waited_on.job.latch.as_ref().unwrap();
    let mut latch_info_lock = latch.info.try_lock().unwrap();
    let waiter = latch_info_lock.waiters.remove(waiter_idx);
    let mut cycle_lock = waiter.cycle.try_lock().unwrap();
    assert!(cycle_lock.is_none());
    *cycle_lock = Some(cycle_error);
    rustc_thread_pool::mark_unblocked(registry);
    waiter.condvar.notify_one();
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

        current_query = query_info.job.parent.map(|i| i.id);
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
