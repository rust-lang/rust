use std::io::Write;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexmap::{self, IndexMap};
use rustc_errors::{Diag, DiagCtxtHandle};
use rustc_hir::def::DefKind;
use rustc_middle::query::{
    CycleError, QueryInfo, QueryJob, QueryJobId, QueryStackDeferred,
    QueryStackFrame,
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
}

#[derive(Clone, Debug)]
pub(crate) struct QueryJobInfo<'tcx> {
    pub(crate) frame: QueryStackFrame<QueryStackDeferred<'tcx>>,
    pub(crate) job: QueryJob<'tcx>,
}

pub(crate) fn find_cycle_in_stack<'tcx>(
    id: QueryJobId,
    job_map: &QueryJobMap<'tcx>,
    mut current_job: Option<QueryJobId>,
    span: Span,
) -> Option<CycleError<QueryStackDeferred<'tcx>>> {
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
            let usage =
                info.job.parent.map(|parent| (info.job.span, job_map.frame_of(parent.id).clone()));
            return Some(CycleError { usage, cycle });
        }

        current_job = info.job.parent.map(|i| i.id);
    }

    None
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

/// Finds a query to break cycle on.
///
/// This function doesn't distinguish between a query wait and a query execution, so both are just
/// query calls.
/// As such some queries may have two or more parent query calls too.
/// It uses depth-first search from a single root query down to the first duplicate query,
/// establishing a cycle.
#[allow(rustc::potential_query_instability)]
fn find_cycle_in_graph<'tcx>(
    query_map: &QueryJobMap<'tcx>,
) -> (QueryJobId, usize, CycleError<QueryStackDeferred<'tcx>>) {
    // We pick any root query we find
    let (&root_query, _) = query_map
        .map
        .iter()
        .find(|(_, info)| info.job.parent.is_none())
        .expect("no root query was found");

    #[derive(Clone, Copy)]
    struct Subquery {
        id: QueryJobId,
        span: Span,
        /// Waiter index or `usize::MAX` if subquery was executed
        waiter_idx: usize,
    }

    // We are allowed to keep track of just one subquery since each query has at least one subquery.
    //
    // If we would assume the opposite then thread of query with no subqueries cannot wait on any
    // subquery. That thread neither can wait on a running parallel task in functions like
    // `par_join`, `par_slice` as the thread executing this parallel task must be blocked too since
    // we are in a deadlock. Rustc only tracks these two cases of blocking code to trigger a
    // deadlock so our assumption has to be false.
    let mut subqueries = FxHashMap::default();
    for query in query_map.map.values() {
        let Some(parent) = query.job.parent else {
            continue;
        };
        // We are safe to only track a single subquery due to the statement above
        subqueries.entry(parent.id).or_insert(Subquery {
            id: query.job.id,
            span: query.job.span,
            waiter_idx: usize::MAX,
        });
    }

    for query in query_map.map.values() {
        let Some(latch) = &query.job.latch else {
            continue;
        };
        // Latch mutexes should be at least about to unlock as we do not hold it anywhere too long
        let lock = latch.info.lock();
        assert!(!lock.complete);
        for (waiter_idx, waiter) in lock.waiters.iter().enumerate() {
            let waited_on_query = waiter.query.expect("cannot wait on a root query");
            // We are safe to only track a single subquery due to the statement above
            subqueries.entry(waited_on_query.id).or_insert(Subquery {
                id: query.job.id,
                span: waiter.span,
                waiter_idx,
            });
        }
    }

    // Debug check the statement above
    if cfg!(debug_assertions) {
        for query in query_map.map.values() {
            assert!(subqueries.contains_key(&query.job.id));
        }
    }

    // At least one thread waits on the first duplicate query in depth-first search stack.
    // Consider this stack of subqueries:
    //
    // ```text
    // a() -> b() -> c() -> b()
    // ```
    //
    // In order for this statement to be false, both occurrences of `b()` only be query executions.
    // Only a single query executes a subquery, so parents of these occurrences of `b()` have to be
    // the same query, aka `a()` and `c()` are equal.
    // However that means `b()` is not the first duplicate query in the stack,
    // so the original statement must be true.
    let mut visited = IndexMap::new();
    let mut last_parent = None;
    let mut last = Subquery { id: root_query, span: DUMMY_SP, waiter_idx: usize::MAX };
    while let indexmap::map::Entry::Vacant(entry) = visited.entry(last.id) {
        entry.insert((last_parent, last));
        last_parent = Some(last.id);
        last = subqueries[&last.id];
    }

    let parent = visited[&last.id].0;
    let mut iter = visited.values();
    let mut cycle = Vec::new();
    loop {
        let (_, subquery) = iter.next_back().unwrap();
        let frame = query_map.map[&subquery.id].frame.clone();
        cycle.push(QueryInfo { span: subquery.span, frame });
        if subquery.id == last.id {
            break;
        }
    }
    cycle.reverse();
    cycle[0].span = last.span;
    let usage = parent.map(|parent| (last.span, query_map.map[&parent].frame.clone()));
    let cycle_error = CycleError { usage, cycle };

    if cfg!(debug_assertions)
        && let Some(expected) = find_cycle_in_stack(last.id, query_map, last_parent, last.span)
    {
        assert!(cycle_error.is_similar_to(&expected));
    }

    // Per statement above we should have wait at either of two occurrences of the duplicate query
    let waiter_idx = if last.waiter_idx != usize::MAX {
        last.waiter_idx
    } else {
        visited.get(&last.id).unwrap().1.waiter_idx
    };

    (last.id, waiter_idx, cycle_error)
}

pub fn break_query_cycles<'tcx>(
    query_map: QueryJobMap<'tcx>,
    registry: &rustc_thread_pool::Registry,
) {
    let (waited_on, waiter_idx, cycle_error) = find_cycle_in_graph(&query_map);

    let waited_on = &query_map.map[&waited_on];
    let latch = waited_on.job.latch.as_ref().unwrap();
    let mut latch_info_lock = latch.info.try_lock().unwrap();

    // And so this `Vec::remove` shouldn't cause a panic
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
