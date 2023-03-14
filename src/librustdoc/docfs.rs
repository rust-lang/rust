//! Rustdoc's FileSystem abstraction module.
//!
//! On Windows this indirects IO into threads to work around performance issues
//! with Defender (and other similar virus scanners that do blocking operations).
//! On other platforms this is a thin shim to fs.
//!
//! Only calls needed to permit this workaround have been abstracted: thus
//! fs::read is still done directly via the fs module; if in future rustdoc
//! needs to read-after-write from a file, then it would be added to this
//! abstraction.

#[cfg(windows)]
use std::cell::RefCell;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::string::ToString;
#[cfg(windows)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
#[cfg(windows)]
use std::sync::Arc;
#[cfg(windows)]
use std::thread::sleep;
#[cfg(windows)]
use std::time::Duration;

pub(crate) trait PathError {
    fn new<S, P: AsRef<Path>>(e: S, path: P) -> Self
    where
        S: ToString + Sized;
}

pub(crate) struct DocFS {
    sync_only: bool,
    errors: Option<Sender<String>>,
    #[cfg(windows)]
    written_files: Arc<AtomicUsize>,
    #[cfg(windows)]
    total_files: RefCell<usize>,
}

impl DocFS {
    pub(crate) fn new(errors: Sender<String>) -> DocFS {
        DocFS {
            sync_only: false,
            errors: Some(errors),
            #[cfg(windows)]
            written_files: Arc::new(AtomicUsize::new(0)),
            #[cfg(windows)]
            total_files: RefCell::new(0),
        }
    }

    pub(crate) fn set_sync_only(&mut self, sync_only: bool) {
        self.sync_only = sync_only;
    }

    pub(crate) fn close(&mut self) {
        self.errors = None;
    }

    pub(crate) fn create_dir_all<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        // For now, dir creation isn't a huge time consideration, do it
        // synchronously, which avoids needing ordering between write() actions
        // and directory creation.
        fs::create_dir_all(path)
    }

    pub(crate) fn write<E>(
        &self,
        path: PathBuf,
        contents: impl 'static + Send + AsRef<[u8]>,
    ) -> Result<(), E>
    where
        E: PathError,
    {
        #[cfg(windows)]
        if !self.sync_only {
            *self.total_files.borrow_mut() += 1;
            // A possible future enhancement after more detailed profiling would
            // be to create the file sync so errors are reported eagerly.
            let sender = self.errors.clone().expect("can't write after closing");
            let written_files = Arc::clone(&self.written_files);
            rayon::spawn(move || {
                fs::write(&path, contents).unwrap_or_else(|e| {
                    sender.send(format!("\"{}\": {}", path.display(), e)).unwrap_or_else(|_| {
                        written_files.fetch_add(1, Ordering::Relaxed);
                        panic!("failed to send error on \"{}\"", path.display())
                    })
                });
                written_files.fetch_add(1, Ordering::Relaxed);
            });
            return;
        }

        fs::write(&path, contents).map_err(|e| E::new(e, path))?;
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for DocFS {
    fn drop(&mut self) {
        let total_files = *self.total_files.borrow();
        while self.written_files.load(Ordering::Relaxed) < total_files {
            sleep(Duration::from_millis(1));
        }
    }
}
