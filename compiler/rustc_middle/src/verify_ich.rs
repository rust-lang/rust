use std::cell::Cell;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_query_system::ich::StableHashingContext;
use tracing::instrument;

use crate::dep_graph::{DepGraphData, SerializedDepNodeIndex};
use crate::ty::TyCtxt;

#[inline]
#[instrument(skip(tcx, dep_graph_data, result, hash_result, format_value), level = "debug")]
pub fn incremental_verify_ich<'tcx, V>(
    tcx: TyCtxt<'tcx>,
    dep_graph_data: &DepGraphData,
    result: &V,
    prev_index: SerializedDepNodeIndex,
    hash_result: Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>,
    format_value: fn(&V) -> String,
) {
    if !dep_graph_data.is_index_green(prev_index) {
        incremental_verify_ich_not_green(tcx, prev_index)
    }

    let new_hash = hash_result.map_or(Fingerprint::ZERO, |f| {
        tcx.with_stable_hashing_context(|mut hcx| f(&mut hcx, result))
    });

    let old_hash = dep_graph_data.prev_fingerprint_of(prev_index);

    if new_hash != old_hash {
        incremental_verify_ich_failed(tcx, prev_index, &|| format_value(result));
    }
}

#[cold]
#[inline(never)]
fn incremental_verify_ich_not_green<'tcx>(tcx: TyCtxt<'tcx>, prev_index: SerializedDepNodeIndex) {
    panic!(
        "fingerprint for green query instance not loaded from cache: {:?}",
        tcx.dep_graph.data().unwrap().prev_node_of(prev_index)
    )
}

// Note that this is marked #[cold] and intentionally takes `dyn Debug` for `result`,
// as we want to avoid generating a bunch of different implementations for LLVM to
// chew on (and filling up the final binary, too).
#[cold]
#[inline(never)]
fn incremental_verify_ich_failed<'tcx>(
    tcx: TyCtxt<'tcx>,
    prev_index: SerializedDepNodeIndex,
    result: &dyn Fn() -> String,
) {
    // When we emit an error message and panic, we try to debug-print the `DepNode`
    // and query result. Unfortunately, this can cause us to run additional queries,
    // which may result in another fingerprint mismatch while we're in the middle
    // of processing this one. To avoid a double-panic (which kills the process
    // before we can print out the query static), we print out a terse
    // but 'safe' message if we detect a reentrant call to this method.
    thread_local! {
        static INSIDE_VERIFY_PANIC: Cell<bool> = const { Cell::new(false) };
    };

    let old_in_panic = INSIDE_VERIFY_PANIC.replace(true);

    if old_in_panic {
        tcx.dcx().emit_err(crate::error::Reentrant);
    } else {
        let run_cmd = if let Some(crate_name) = &tcx.sess.opts.crate_name {
            format!("`cargo clean -p {crate_name}` or `cargo clean`")
        } else {
            "`cargo clean`".to_string()
        };

        let dep_node = tcx.dep_graph.data().unwrap().prev_node_of(prev_index);
        tcx.dcx().emit_err(crate::error::IncrementCompilation {
            run_cmd,
            dep_node: format!("{dep_node:?}"),
        });
        panic!("Found unstable fingerprints for {dep_node:?}: {}", result());
    }

    INSIDE_VERIFY_PANIC.set(old_in_panic);
}
