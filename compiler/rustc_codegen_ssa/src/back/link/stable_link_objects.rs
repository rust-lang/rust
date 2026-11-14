use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rustc_fs_util::TempDirBuilder;
use rustc_session::Session;

// Both linker inputs have stable names, protected by the same per-output lock.
// The lockfile must not be removed: another invocation could otherwise lock a
// different inode while an earlier invocation still holds the original lock.
pub(super) struct StableLinkObjects {
    pub(super) metadata_path: PathBuf,
    pub(super) symbols_path: PathBuf,
    _lock: Option<fs::File>,
    keep: bool,
    pub(super) metadata_created: std::cell::Cell<bool>,
    pub(super) symbols_created: std::cell::Cell<bool>,
}

impl Drop for StableLinkObjects {
    fn drop(&mut self) {
        if !self.keep {
            if self.metadata_created.get() {
                let _ = fs::remove_file(&self.metadata_path);
            }
            if self.symbols_created.get() {
                let _ = fs::remove_file(&self.symbols_path);
            }
        }
    }
}

// Create a linker input under the per-output lock. A previous invocation may
// have crashed, leaving a stale object. Never follow a pre-existing symlink.
pub(super) fn write_stable_link_object(
    sess: &Session,
    path: &Path,
    data: &[u8],
    created: &std::cell::Cell<bool>,
) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => sess
            .dcx()
            .fatal(format!("failed to remove stale linker object {}: {e}", path.display())),
    }

    let mut file =
        fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap_or_else(|e| {
            sess.dcx().fatal(format!("failed to create linker object {}: {e}", path.display()))
        });

    // Cleanup must also cover a partially written object.
    created.set(true);
    file.write_all(data).unwrap_or_else(|e| {
        sess.dcx().fatal(format!("failed to write linker object {}: {e}", path.display()))
    });
}

impl StableLinkObjects {
    pub(super) fn new(
        sess: &Session,
        out_filename: &Path,
        needs_metadata: bool,
        needs_symbols: bool,
    ) -> Self {
        // Lock both stable linker inputs together, from object creation until
        // the linker and any post-link processing have finished.
        let stable_path = |suffix: &str| {
            let mut name = out_filename.as_os_str().to_os_string();
            name.push(suffix);
            PathBuf::from(name)
        };

        // Keep linker inputs in a dedicated, deterministic namespace.
        let objects_dir = stable_path(".rustc-link-objects");
        let metadata_path = objects_dir.join("metadata.o");
        let symbols_path = objects_dir.join("symbols.o");

        let lock = if needs_metadata || needs_symbols {
            // Never unlink this lockfile: doing so could split the lock
            // across different inodes.
            let lock_path = stable_path(".rustc-link-objects.lock");
            let lock = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&lock_path)
                .unwrap_or_else(|e| {
                    sess.dcx().fatal(format!("failed to open {}: {e}", lock_path.display()))
                });
            lock.lock().unwrap_or_else(|e| {
                sess.dcx().fatal(format!("failed to lock {}: {e}", lock_path.display()))
            });

            const MARKER: &[u8] = b"rustc-link-objects-v1\n";
            let marker_path = objects_dir.join("owner");

            match fs::symlink_metadata(&objects_dir) {
                Ok(meta) if meta.file_type().is_dir() => {}
                Ok(_) => sess.dcx().fatal(format!(
                    "linker object directory {} is not a directory",
                    objects_dir.display()
                )),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    // Prepare the ownership marker before publishing the
                    // directory. An interrupted initialization must not
                    // leave an incomplete directory at the stable path.
                    let parent = objects_dir.parent().unwrap_or_else(|| Path::new("."));
                    let staging = TempDirBuilder::new()
                        .prefix("rustc-link-objects-")
                        .tempdir_in(parent)
                        .unwrap_or_else(|e| {
                            sess.dcx().fatal(format!(
                                "failed to create linker object staging directory: {e}"
                            ))
                        });

                    fs::write(staging.path().join("owner"), MARKER).unwrap_or_else(|e| {
                        sess.dcx()
                            .fatal(format!("failed to initialize linker object directory: {e}"))
                    });

                    fs::rename(staging.path(), &objects_dir).unwrap_or_else(|e| {
                        sess.dcx()
                            .fatal(format!("failed to initialize {}: {e}", objects_dir.display()))
                    });
                }
                Err(e) => {
                    sess.dcx().fatal(format!("failed to inspect {}: {e}", objects_dir.display()))
                }
            }

            // Do not adopt unrelated directories.
            let valid_marker = fs::symlink_metadata(&marker_path)
                .is_ok_and(|meta| meta.file_type().is_file())
                && fs::read(&marker_path).is_ok_and(|data| data == MARKER);

            if !valid_marker {
                sess.dcx().fatal(format!(
                    "unrecognized linker object directory {}",
                    objects_dir.display()
                ));
            }

            Some(lock)
        } else {
            None
        };

        let stable_objects = StableLinkObjects {
            metadata_path,
            symbols_path,
            _lock: lock,
            keep: sess.opts.cg.save_temps,
            metadata_created: std::cell::Cell::new(false),
            symbols_created: std::cell::Cell::new(false),
        };

        stable_objects
    }
}
