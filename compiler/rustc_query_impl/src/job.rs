use std::io::Write;

use rustc_data_structures::hash_table::HashTable;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_middle::query::q_impl::job::{CollectActiveJobsKind, QueryJobInfo, QueryJobMap};
use rustc_middle::query::{ActiveKeyStatus, QueryCache, QueryJobId, QueryKey, QueryVTable};
use rustc_middle::ty::TyCtxt;
use tracing::debug;

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

/// Internal plumbing for collecting the set of active jobs for this query.
///
/// Aborts if jobs can't be gathered as specified by `collect_kind`.
fn collect_active_query_jobs_inner<'tcx, C>(
    query: &'tcx QueryVTable<'tcx, C>,
    collect_kind: CollectActiveJobsKind,
    job_map: &mut QueryJobMap<'tcx>,
) where
    C: QueryCache<Key: QueryKey + DynSend + DynSync>,
    QueryVTable<'tcx, C>: DynSync,
{
    let mut collect_shard_jobs = |shard: &HashTable<(C::Key, ActiveKeyStatus<'tcx>)>| {
        for (key, status) in shard.iter() {
            if let ActiveKeyStatus::Started(job) = status {
                // It's fine to call `create_tagged_key` with the shard locked,
                // because it's just a `TaggedQueryKey` variant constructor.
                let tagged_key = (query.create_tagged_key)(*key);
                job_map.insert(job.id, QueryJobInfo { tagged_key, job: job.clone() });
            }
        }
    };

    match collect_kind {
        CollectActiveJobsKind::Full => {
            for shard in query.state.active.lock_shards() {
                collect_shard_jobs(&shard);
            }
        }
        CollectActiveJobsKind::FullNoContention => {
            for shard in query.state.active.try_lock_shards() {
                match shard {
                    Some(shard) => collect_shard_jobs(&shard),
                    None => panic!("Failed to collect active jobs for query `{}`!", query.name),
                }
            }
        }
        CollectActiveJobsKind::PartialAllowed => {
            for shard in query.state.active.try_lock_shards() {
                match shard {
                    Some(shard) => collect_shard_jobs(&shard),
                    // This collection is best-effort (it is only used to print the query
                    // stack on panic), so a contended shard is expected and fine to skip.
                    // Emitting this at `warn!` would leak nondeterministically into the
                    // panic output under the parallel front-end, where another thread may
                    // still hold a shard lock, so keep it at `debug!`.
                    None => debug!("Failed to collect active jobs for query `{}`!", query.name),
                }
            }
        }
    }
}

pub fn print_query_stack<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut current_query: Option<QueryJobId>,
    limit_frames: Option<usize>,
    mut file: Option<std::fs::File>,
) -> usize {
    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `DiagCtxt` may be in a weird
    // state if it was responsible for triggering the panic.
    let mut count_printed = 0;
    let mut count_total = 0;

    // Make use of a partial query job map if we fail to take locks collecting active queries.
    let job_map = tcx.collect_active_query_jobs(CollectActiveJobsKind::PartialAllowed);

    if let Some(ref mut file) = file {
        let _ = writeln!(file, "\n\nquery stack during panic:");
    }
    while let Some(query) = current_query {
        let Some(query_info) = job_map.info(query) else {
            break;
        };
        let description = query_info.tagged_key.description(tcx);
        if Some(count_printed) < limit_frames || limit_frames.is_none() {
            // Only print to stderr as many stack frames as `num_frames` when present.
            eprintln!(
                "#{count_printed} [{query_name}] {description}",
                query_name = query_info.tagged_key.query_name(),
            );
            count_printed += 1;
        }

        if let Some(ref mut file) = file {
            let _ = writeln!(
                file,
                "#{count_total} [{query_name}] {description}",
                query_name = query_info.tagged_key.query_name(),
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
