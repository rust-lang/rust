use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;
use tracing::instrument;

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_fetch_file_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: Symbol,
) -> Option<&'tcx [u8]> {
    let Some(crate_root_path) = tcx.sess.io.input.opt_path() else {
        return None;
    };

    let requested_path = key.as_str();

    // Always resolve path from the crate root to provide consistent behavior.
    let resolved_path = crate_root_path.with_file_name(requested_path);

    // Very simple implementation where we whitelist paths directly. We don't even attempt
    // to canonicalize for now.
    if tcx
        .sess
        .opts
        .unstable_opts
        .experimental_comptime_files
        .iter()
        .find(|s| requested_path == *s)
        .is_some()
    {
        match std::fs::read(resolved_path) {
            Ok(content) => Some(tcx.arena.alloc_slice(&content)),
            Err(err) => {
                tracing::debug!(%err, "failed to read file");
                None
            }
        }
    } else {
        tracing::debug!("not allowed to read file");
        None
    }
}
