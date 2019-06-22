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

use errors;

use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, Sender};

macro_rules! try_err {
    ($e:expr, $file:expr) => {{
        match $e {
            Ok(e) => e,
            Err(e) => return Err(E::new(e, $file)),
        }
    }};
}

pub trait PathError {
    fn new<P: AsRef<Path>>(e: io::Error, path: P) -> Self;
}

pub struct ErrorStorage {
    sender: Option<Sender<Option<String>>>,
    receiver: Receiver<Option<String>>,
}

impl ErrorStorage {
    pub fn new() -> ErrorStorage {
        let (sender, receiver) = channel();
        ErrorStorage {
            sender: Some(sender),
            receiver,
        }
    }

    /// Prints all stored errors. Returns the number of printed errors.
    pub fn write_errors(&mut self, diag: &errors::Handler) -> usize {
        let mut printed = 0;
        // In order to drop the sender part of the channel.
        self.sender = None;

        for msg in self.receiver.iter() {
            if let Some(ref error) = msg {
                diag.struct_err(&error).emit();
                printed += 1;
            }
        }
        printed
    }
}

pub struct DocFS {
    sync_only: bool,
    errors: Arc<ErrorStorage>,
}

impl DocFS {
    pub fn new(errors: &Arc<ErrorStorage>) -> DocFS {
        DocFS {
            sync_only: false,
            errors: Arc::clone(errors),
        }
    }

    pub fn set_sync_only(&mut self, sync_only: bool) {
        self.sync_only = sync_only;
    }

    pub fn create_dir_all<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        // For now, dir creation isn't a huge time consideration, do it
        // synchronously, which avoids needing ordering between write() actions
        // and directory creation.
        fs::create_dir_all(path)
    }

    pub fn write<P, C, E>(&self, path: P, contents: C) -> Result<(), E>
    where
        P: AsRef<Path>,
        C: AsRef<[u8]>,
        E: PathError,
    {
        if !self.sync_only && cfg!(windows) {
            // A possible future enhancement after more detailed profiling would
            // be to create the file sync so errors are reported eagerly.
            let contents = contents.as_ref().to_vec();
            let path = path.as_ref().to_path_buf();
            let sender = self.errors.sender.clone().unwrap();
            rayon::spawn(move || {
                match fs::write(&path, &contents) {
                    Ok(_) => {
                        sender.send(None)
                            .expect(&format!("failed to send error on \"{}\"", path.display()));
                    }
                    Err(e) => {
                        sender.send(Some(format!("\"{}\": {}", path.display(), e)))
                            .expect(&format!("failed to send non-error on \"{}\"", path.display()));
                    }
                }
            });
            Ok(())
        } else {
            Ok(try_err!(fs::write(&path, contents), path))
        }
    }
}
