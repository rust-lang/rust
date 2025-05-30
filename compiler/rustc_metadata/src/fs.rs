use std::path::{Path, PathBuf};
use std::{fs, io};

use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_fs_util::TempDirBuilder;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{CrateType, OutFileName, OutputType};
use rustc_session::output::filename_for_metadata;

use crate::errors::{
    BinaryOutputToTty, FailedCopyToStdout, FailedCreateEncodedMetadata, FailedCreateFile,
    FailedCreateTempdir, FailedWriteError,
};
use crate::{EncodedMetadata, encode_metadata};

// FIXME(eddyb) maybe include the crate name in this?
pub const METADATA_FILENAME: &str = "lib.rmeta";

/// We use a temp directory here to avoid races between concurrent rustc processes,
/// such as builds in the same directory using the same filename for metadata while
/// building an `.rlib` (stomping over one another), or writing an `.rmeta` into a
/// directory being searched for `extern crate` (observing an incomplete file).
/// The returned path is the temporary file containing the complete metadata.
pub fn emit_wrapper_file(sess: &Session, data: &[u8], tmpdir: &Path, name: &str) -> PathBuf {
    let out_filename = tmpdir.join(name);
    let result = fs::write(&out_filename, data);

    if let Err(err) = result {
        sess.dcx().emit_fatal(FailedWriteError { filename: out_filename, err });
    }

    out_filename
}

pub fn encode_and_write_metadata(tcx: TyCtxt<'_>) -> EncodedMetadata {
    let out_filename = filename_for_metadata(tcx.sess, tcx.output_filenames(()));
    // To avoid races with another rustc process scanning the output directory,
    // we need to write the file somewhere else and atomically move it to its
    // final destination, with an `fs::rename` call. In order for the rename to
    // always succeed, the temporary file needs to be on the same filesystem,
    // which is why we create it inside the output directory specifically.
    let metadata_tmpdir = TempDirBuilder::new()
        .prefix("rmeta")
        .tempdir_in(out_filename.parent().unwrap_or_else(|| Path::new("")))
        .unwrap_or_else(|err| tcx.dcx().emit_fatal(FailedCreateTempdir { err }));
    let metadata_tmpdir = MaybeTempDir::new(metadata_tmpdir, tcx.sess.opts.cg.save_temps);
    let metadata_filename = metadata_tmpdir.as_ref().join("full.rmeta");
    let metadata_stub_filename = if !tcx.sess.opts.unstable_opts.embed_metadata
        && !tcx.crate_types().contains(&CrateType::ProcMacro)
    {
        Some(metadata_tmpdir.as_ref().join("stub.rmeta"))
    } else {
        None
    };

    if tcx.needs_metadata() {
        encode_metadata(tcx, &metadata_filename, metadata_stub_filename.as_deref());
    } else {
        // Always create a file at `metadata_filename`, even if we have nothing to write to it.
        // This simplifies the creation of the output `out_filename` when requested.
        std::fs::File::create(&metadata_filename).unwrap_or_else(|err| {
            tcx.dcx().emit_fatal(FailedCreateFile { filename: &metadata_filename, err });
        });
        if let Some(metadata_stub_filename) = &metadata_stub_filename {
            std::fs::File::create(metadata_stub_filename).unwrap_or_else(|err| {
                tcx.dcx().emit_fatal(FailedCreateFile { filename: &metadata_stub_filename, err });
            });
        }
    }

    let _prof_timer = tcx.sess.prof.generic_activity("write_crate_metadata");

    // If the user requests metadata as output, rename `metadata_filename`
    // to the expected output `out_filename`. The match above should ensure
    // this file always exists.
    let need_metadata_file = tcx.sess.opts.output_types.contains_key(&OutputType::Metadata);
    let (metadata_filename, metadata_tmpdir) = if need_metadata_file {
        let filename = match out_filename {
            OutFileName::Real(ref path) => {
                if let Err(err) = non_durable_rename(&metadata_filename, path) {
                    tcx.dcx().emit_fatal(FailedWriteError { filename: path.to_path_buf(), err });
                }
                path.clone()
            }
            OutFileName::Stdout => {
                if out_filename.is_tty() {
                    tcx.dcx().emit_err(BinaryOutputToTty);
                } else if let Err(err) = copy_to_stdout(&metadata_filename) {
                    tcx.dcx()
                        .emit_err(FailedCopyToStdout { filename: metadata_filename.clone(), err });
                }
                metadata_filename
            }
        };
        if tcx.sess.opts.json_artifact_notifications {
            tcx.dcx().emit_artifact_notification(out_filename.as_path(), "metadata");
        }
        (filename, None)
    } else {
        (metadata_filename, Some(metadata_tmpdir))
    };

    // Load metadata back to memory: codegen may need to include it in object files.
    let metadata =
        EncodedMetadata::from_path(metadata_filename, metadata_stub_filename, metadata_tmpdir)
            .unwrap_or_else(|err| {
                tcx.dcx().emit_fatal(FailedCreateEncodedMetadata { err });
            });

    metadata
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

pub fn copy_to_stdout(from: &Path) -> io::Result<()> {
    let mut reader = fs::File::open_buffered(from)?;
    let mut stdout = io::stdout();
    io::copy(&mut reader, &mut stdout)?;
    Ok(())
}
