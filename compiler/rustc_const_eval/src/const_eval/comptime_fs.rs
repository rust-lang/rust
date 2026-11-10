use std::io::ErrorKind;

use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::comptime::ReadFileError;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;
use tracing::instrument;

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_list_files_provider(tcx: TyCtxt<'_>, _: LocalCrate) -> &[Symbol] {
    let filelist = tcx
        .sess
        .opts
        .unstable_opts
        .experimental_comptime_files
        .iter()
        .map(|p| Symbol::intern(p))
        .collect::<Vec<_>>();

    tcx.arena.alloc_slice(&filelist)
}

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_fetch_file_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: Symbol,
) -> Result<Option<&'tcx [u8]>, ReadFileError<'tcx>> {
    // In case we are in a context without a crate root, just return that
    // every file is whitelisted but does not exist. Is this code path
    // taken by rustdoc when checking a comment code block?
    let Some(crate_root_path) = tcx.sess.io.input.opt_path() else {
        return Ok(None);
    };

    let requested_path = key.as_str();

    // Always resolve path from the crate root to provide consistent behavior.
    let resolved_path = crate_root_path.with_file_name(requested_path);

    // Very simple implementation where we whitelist paths directly. We don't even attempt
    // to canonicalize for now.
    if tcx
        .experimental_comptime_list_files(LOCAL_CRATE)
        .iter()
        .find(|s| requested_path == s.as_str())
        .is_some()
    {
        match std::fs::read(resolved_path) {
            Ok(content) => Ok(Some(tcx.arena.alloc_slice(&content))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ReadFileError::IO(tcx.arena.alloc_str(&e.to_string()))),
        }
    } else {
        Err(ReadFileError::NotWhitelisted)
    }
}
