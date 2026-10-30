use rustc_middle::ty::TyCtxt;
use tracing::instrument;

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_fetch_file_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: &'tcx String,
) -> Option<&'tcx [u8]> {
    // Very simple implementation where we whitelist paths directly. We don't even attempt
    // to canonilize for now.
    if tcx.sess.opts.unstable_opts.experimental_comptime_files.contains(key) {
        let file = std::fs::read(key).ok()?;
        Some(&*tcx.arena.alloc_slice(&file))
    } else {
        None
    }
}
