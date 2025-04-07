use std::ops::Deref as _;
use std::path::{Path, PathBuf};
use std::{fs, io};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{InstanceKind, TyCtxt};
use rustc_session::config::{CrateType, OutFileName, OutputType};
use rustc_session::output::filename_for_metadata;
use rustc_session::{MetadataKind, Session};
use tempfile::Builder as TempFileBuilder;

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
pub fn emit_wrapper_file(
    sess: &Session,
    data: &[u8],
    tmpdir: &MaybeTempDir,
    name: &str,
) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(name);
    let result = fs::write(&out_filename, data);

    if let Err(err) = result {
        sess.dcx().emit_fatal(FailedWriteError { filename: out_filename, err });
    }

    out_filename
}

pub fn encode_and_write_metadata(tcx: TyCtxt<'_>) -> (EncodedMetadata, bool) {
    let out_filename = filename_for_metadata(tcx.sess, tcx.output_filenames(()));
    //let hash = tcx.crate_hash()
    // To avoid races with another rustc process scanning the output directory,
    // we need to write the file somewhere else and atomically move it to its
    // final destination, with an `fs::rename` call. In order for the rename to
    // always succeed, the temporary file needs to be on the same filesystem,
    // which is why we create it inside the output directory specifically.
    let metadata_tmpdir = TempFileBuilder::new()
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

    // Always create a file at `metadata_filename`, even if we have nothing to write to it.
    // This simplifies the creation of the output `out_filename` when requested.
    let metadata_kind = tcx.metadata_kind();
    match metadata_kind {
        MetadataKind::None => {
            std::fs::File::create(&metadata_filename).unwrap_or_else(|err| {
                tcx.dcx().emit_fatal(FailedCreateFile { filename: &metadata_filename, err });
            });
            if let Some(metadata_stub_filename) = &metadata_stub_filename {
                std::fs::File::create(metadata_stub_filename).unwrap_or_else(|err| {
                    tcx.dcx()
                        .emit_fatal(FailedCreateFile { filename: &metadata_stub_filename, err });
                });
            }
        }
        MetadataKind::Uncompressed | MetadataKind::Compressed => {
            encode_metadata(tcx, &metadata_filename, metadata_stub_filename.as_deref())
        }
    };

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
            let hash = public_api_hash(tcx);
            tcx.dcx().emit_artifact_notification(
                out_filename.as_path(),
                "metadata",
                Some(&hash.to_string()),
            );
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

pub fn copy_to_stdout(from: &Path) -> io::Result<()> {
    let mut reader = fs::File::open_buffered(from)?;
    let mut stdout = io::stdout();
    io::copy(&mut reader, &mut stdout)?;
    Ok(())
}

fn public_api_hash(tcx: TyCtxt<'_>) -> Fingerprint {
    let mut stable_hasher = StableHasher::new();
    tcx.with_stable_hashing_context(|mut hcx| {
        hcx.while_hashing_spans(false, |mut hcx| {
            let _ = tcx
                .reachable_set(())
                .to_sorted(hcx, true)
                .into_iter()
                .filter_map(|local_def_id: &LocalDefId| {
                    let def_id = local_def_id.to_def_id();

                    let item = tcx.hir_node_by_def_id(*local_def_id);
                    let _ = item.ident()?;
                    let has_hir_body = item.body_id().is_some();

                    let item_path = tcx.def_path(def_id);
                    let def_kind = tcx.def_kind(def_id);
                    let mut fn_sig = None;
                    item_path.to_string_no_crate_verbose().hash_stable(hcx, &mut stable_hasher);
                    let has_mir = match def_kind {
                        DefKind::Ctor(_, _)
                        | DefKind::AnonConst
                        | DefKind::InlineConst
                        | DefKind::AssocConst
                        | DefKind::Const
                        | DefKind::SyntheticCoroutineBody => has_hir_body,
                        DefKind::AssocFn | DefKind::Fn | DefKind::Closure => {
                            fn_sig = Some(tcx.fn_sig(def_id));
                            if def_kind == DefKind::Closure && tcx.is_coroutine(def_id) {
                                has_hir_body
                            } else {
                                let generics = tcx.generics_of(def_id);
                                has_hir_body
                                    && (tcx.sess.opts.unstable_opts.always_encode_mir
                                        || (tcx.sess.opts.output_types.should_codegen()
                                            && (generics.requires_monomorphization(tcx)
                                                || tcx.cross_crate_inlinable(def_id))))
                            }
                        }
                        _ => {
                            return None;
                        }
                    };

                    if let Some(sig) = fn_sig {
                        sig.skip_binder().hash_stable(hcx, &mut stable_hasher);
                    }
                    if !has_mir {
                        return Some(());
                    }

                    let ty = tcx.type_of(def_id);

                    let body = tcx.instance_mir(InstanceKind::Item(def_id));
                    let blocks = body.basic_blocks.deref();

                    // Deref to avoid hashing cache of mir body.
                    let _ = blocks
                        .iter()
                        .map(|bb| {
                            let kind =
                                bb.terminator.as_ref().map(|terminator| terminator.kind.clone());
                            let statements = bb
                                .statements
                                .iter()
                                .map(|statement| statement.kind.clone())
                                .collect::<Vec<_>>();

                            (bb.is_cleanup, kind, statements)
                                .hash_stable(&mut hcx, &mut stable_hasher);
                            ()
                        })
                        .collect::<Vec<_>>();

                    ty.skip_binder().kind().hash_stable(&mut hcx, &mut stable_hasher);

                    Some(())
                })
                .collect::<Vec<_>>();
        });
        stable_hasher.finish()
    })
}
