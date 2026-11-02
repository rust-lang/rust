use rustc_middle::query::q_impl::job::{
    CollectActiveJobsKind, QueryJobMap, collect_active_query_jobs_inner,
};
use rustc_middle::query::{QueryCycle, QueryJobId, QueryStackFrame};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::query_vtables::for_each_query_vtable;

/// Returns a map of currently active query jobs, collected from all queries.
pub fn collect_active_query_jobs<'tcx>(
    tcx: TyCtxt<'tcx>,
    collect_kind: CollectActiveJobsKind,
) -> QueryJobMap<'tcx> {
    let mut job_map = QueryJobMap::default();

    for_each_query_vtable!(ALL, tcx, |query| {
        collect_active_query_jobs_inner(query, collect_kind, &mut job_map);
    });

    job_map
}

pub(crate) fn find_cycle_in_stack<'tcx>(
    id: QueryJobId,
    job_map: QueryJobMap<'tcx>,
    current_job: &Option<QueryJobId>,
    span: Span,
) -> QueryCycle<'tcx> {
    // Find the waitee amongst `current_job` parents.
    let mut frames = Vec::new();
    let mut current_job = Option::clone(current_job);

    while let Some(job) = current_job {
        let info = job_map.info(job);
        frames.push(QueryStackFrame { span: info.job.span, tagged_key: info.tagged_key });

        if job == id {
            frames.reverse();

            // This is the end of the cycle. The span entry we included was for
            // the usage of the cycle itself, and not part of the cycle.
            // Replace it with the span which caused the cycle to form.
            frames[0].span = span;
            // Find out why the cycle itself was used.
            let usage = try {
                let parent = info.job.parent?;
                QueryStackFrame { span: info.job.span, tagged_key: job_map.tagged_key_of(parent) }
            };
            return QueryCycle { usage, frames };
        }

        current_job = info.job.parent;
    }

    panic!("did not find a cycle")
}
