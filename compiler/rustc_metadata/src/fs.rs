use crate::{encode_metadata, EncodedMetadata};

use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{CrateType, OutputFilenames, OutputType};
use rustc_session::output::filename_for_metadata;
use rustc_session::Session;
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
pub fn emit_metadata(sess: &Session, metadata: &[u8], tmpdir: &MaybeTempDir) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(METADATA_FILENAME);
    let result = fs::write(&out_filename, metadata);

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }

    out_filename
}

pub fn encode_and_write_metadata(
    tcx: TyCtxt<'_>,
    outputs: &OutputFilenames,
) -> (EncodedMetadata, bool) {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed,
    }

    let metadata_kind = tcx
        .sess
        .crate_types()
        .iter()
        .map(|ty| match *ty {
            CrateType::Executable | CrateType::Staticlib | CrateType::Cdylib => MetadataKind::None,

            CrateType::Rlib => MetadataKind::Uncompressed,

            CrateType::Dylib | CrateType::ProcMacro => MetadataKind::Compressed,
        })
        .max()
        .unwrap_or(MetadataKind::None);

    let crate_name = tcx.crate_name(LOCAL_CRATE);
    let out_filename = filename_for_metadata(tcx.sess, crate_name.as_str(), outputs);
    // To avoid races with another rustc process scanning the output directory,
    // we need to write the file somewhere else and atomically move it to its
    // final destination, with an `fs::rename` call. In order for the rename to
    // always succeed, the temporary file needs to be on the same filesystem,
    // which is why we create it inside the output directory specifically.
    let metadata_tmpdir = TempFileBuilder::new()
        .prefix("rmeta")
        .tempdir_in(out_filename.parent().unwrap_or_else(|| Path::new("")))
        .unwrap_or_else(|err| tcx.sess.fatal(&format!("couldn't create a temp dir: {}", err)));
    let metadata_tmpdir = MaybeTempDir::new(metadata_tmpdir, tcx.sess.opts.cg.save_temps);
    let metadata_filename = metadata_tmpdir.as_ref().join(METADATA_FILENAME);
    let metadata = match metadata_kind {
        MetadataKind::None => EncodedMetadata::new(),
        MetadataKind::Uncompressed | MetadataKind::Compressed => {
            encode_metadata(tcx, metadata_filename)
        }
    };

    let _prof_timer = tcx.sess.prof.generic_activity("write_crate_metadata");

    let need_metadata_file = tcx.sess.opts.output_types.contains_key(&OutputType::Metadata);
    if need_metadata_file {
        let metadata_filename = emit_metadata(tcx.sess, metadata.raw_data(), &metadata_tmpdir);
        if let Err(e) = non_durable_rename(&metadata_filename, &out_filename) {
            tcx.sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
        }
        if tcx.sess.opts.json_artifact_notifications {
            tcx.sess
                .parse_sess
                .span_diagnostic
                .emit_artifact_notification(&out_filename, "metadata");
        }
    }

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
