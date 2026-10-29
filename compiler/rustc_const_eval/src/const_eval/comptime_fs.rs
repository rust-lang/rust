use rustc_middle::ty::TyCtxt;
use tracing::instrument;

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_fetch_file_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: &'tcx String,
) -> Option<&'tcx [u8]> {
    let mut files = tcx.caches.experimental_comptime_files.lock();
    if let Some(value) = files.get(&key) {
        Some(*value)
    } else {
        let file = std::fs::read(key).ok()?;
        files.insert(&key, tcx.arena.alloc_slice(&file));
        files.get(&key).map(|d| *d)
    }
}
