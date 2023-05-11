use crate::errors::{
    FailedCreateEncodedMetadata, FailedCreateFile, FailedCreateTempdir, FailedWriteError,
};
use crate::{encode_metadata, EncodedMetadata};

use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OutputType;
use rustc_session::output::filename_for_metadata;
use rustc_session::{MetadataKind, Session};
use tempfile::Builder as TempFileBuilder;

use std::fs;
use std::path::{Path, PathBuf};

// FIXME(eddyb) maybe include the crate name in this?
pub const METADATA_FILENAME: &str = "lib.rmeta";

/// We use a temp directory here to avoid races between concurrent rustc processes,
/// such as builds in the same directory using the same filename for metadata while
/// building an `.rlib` (stomping over one another), or writing an `.rmeta` into a
/// directory being searched for `extern crate` (observing an incomplete file).
/// The returned path is the temporary file containing the complete metadata.
pub fn emit_wrapper_file(
    sess: &Session,
    data: &[u8],
    tmpdir: &MaybeTempDir,
    name: &str,
) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(name);
    let result = fs::write(&out_filename, data);

    if let Err(err) = result {
        sess.emit_fatal(FailedWriteError { filename: out_filename, err });
    }

    out_filename
}

pub fn encode_and_write_metadata(tcx: TyCtxt<'_>) -> (EncodedMetadata, bool) {
    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let out_filename = filename_for_metadata(tcx.sess, crate_name, tcx.output_filenames(()));
    // To avoid races with another rustc process scanning the output directory,
    // we need to write the file somewhere else and atomically move it to its
    // final destination, with an `fs::rename` call. In order for the rename to
    // always succeed, the temporary file needs to be on the same filesystem,
    // which is why we create it inside the output directory specifically.
    let metadata_tmpdir = TempFileBuilder::new()
        .prefix("rmeta")
        .tempdir_in(out_filename.parent().unwrap_or_else(|| Path::new("")))
        .unwrap_or_else(|err| tcx.sess.emit_fatal(FailedCreateTempdir { err }));
    let metadata_tmpdir = MaybeTempDir::new(metadata_tmpdir, tcx.sess.opts.cg.save_temps);
    let metadata_filename = metadata_tmpdir.as_ref().join(METADATA_FILENAME);

    // Always create a file at `metadata_filename`, even if we have nothing to write to it.
    // This simplifies the creation of the output `out_filename` when requested.
    let metadata_kind = tcx.sess.metadata_kind();
    match metadata_kind {
        MetadataKind::None => {
            std::fs::File::create(&metadata_filename).unwrap_or_else(|err| {
                tcx.sess.emit_fatal(FailedCreateFile { filename: &metadata_filename, err });
            });
        }
        MetadataKind::Uncompressed | MetadataKind::Compressed => {
            encode_metadata(tcx, &metadata_filename);
        }
    };

    let _prof_timer = tcx.sess.prof.generic_activity("write_crate_metadata");

    // If the user requests metadata as output, rename `metadata_filename`
    // to the expected output `out_filename`. The match above should ensure
    // this file always exists.
    let need_metadata_file = tcx.sess.opts.output_types.contains_key(&OutputType::Metadata);
    let (metadata_filename, metadata_tmpdir) = if need_metadata_file {
        if let Err(err) = non_durable_rename(&metadata_filename, &out_filename) {
            tcx.sess.emit_fatal(FailedWriteError { filename: out_filename, err });
        }
        if tcx.sess.opts.json_artifact_notifications {
            tcx.sess
                .parse_sess
                .span_diagnostic
                .emit_artifact_notification(&out_filename, "metadata");
        }
        (out_filename, None)
    } else {
        (metadata_filename, Some(metadata_tmpdir))
    };

    // Load metadata back to memory: codegen may need to include it in object files.
    let metadata =
        EncodedMetadata::from_path(metadata_filename, metadata_tmpdir).unwrap_or_else(|err| {
            tcx.sess.emit_fatal(FailedCreateEncodedMetadata { err });
        });

    let need_metadata_module = metadata_kind == MetadataKind::Compressed;

    (metadata, need_metadata_module)
}

#[cfg(not(target_os = "linux"))]
pub fn non_durable_rename(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::rename(src, dst)
}

/// This function attempts to bypass the auto_da_alloc heuristic implemented by some filesystems
/// such as btrfs and ext4. When renaming over a file that already exists then they will "helpfully"
/// write back the source file before committing the rename in case a developer forgot some of
/// the fsyncs in the open/write/fsync(file)/rename/fsync(dir) dance for atomic file updates.
///
/// To avoid triggering this heuristic we delete the destination first, if it exists.
/// The cost of an extra syscall is much lower than getting descheduled for the sync IO.
#[cfg(target_os = "linux")]
pub fn non_durable_rename(src: &Path, dst: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(dst);
    std::fs::rename(src, dst)
}
