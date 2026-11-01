use rustc_middle::query::q_impl::job::{
    CollectActiveJobsKind, QueryJobMap, collect_active_query_jobs_inner,
};
use rustc_middle::ty::TyCtxt;

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
