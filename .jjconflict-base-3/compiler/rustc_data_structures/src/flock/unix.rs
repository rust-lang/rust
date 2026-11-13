use std::collections::hash_map::Entry;
use std::fs::File;
use std::os::unix::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::{io, mem};

use rustc_hash::FxHashMap;

static LOCK_REGISTRY: LazyLock<Mutex<FxHashMap<PathBuf, LockState>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

enum LockState {
    /// Lock exclusively held. `Lock.file` contains an `Arc<UnlockGuard>` with a single reference.
    ///
    /// The `extra_files` fields contains files we opened while the lock was held. We have to
    /// persist them until we actually want to unlock the file to prevent unlocking on close.
    Exclusive { extra_files: Vec<File> },
    /// Lock can be shared. When there are N lock holders, the `Arc<UnlockGuard>` has N+1 references
    /// with the last one being held by `LockState` and getting removed in the drop impl of `Lock`
    /// if it is the remaining reference.
    Shared(Arc<UnlockGuard>),
}

#[derive(Debug)]
pub struct Lock {
    path: PathBuf,
    file: Option<Arc<UnlockGuard>>,
}

impl Lock {
    pub fn try_lock(p: &Path, file: File, exclusive: bool) -> io::Result<Lock> {
        let mut locks = LOCK_REGISTRY.lock().unwrap();

        let file = match locks.entry(p.to_owned()) {
            Entry::Occupied(mut state) => {
                // We must not open the file again if there is an existing lock to prevent the close
                // from unlocking the file even when another `Lock` already had the lock held before
                // this `Lock::try_lock` call.
                match state.get_mut() {
                    LockState::Exclusive { extra_files } => {
                        // Retain file to prevent unlock on close
                        extra_files.push(file);

                        return Err(io::ErrorKind::WouldBlock.into());
                    }
                    LockState::Shared(file) => {
                        if exclusive {
                            return Err(io::ErrorKind::WouldBlock.into());
                        } else {
                            Arc::clone(file)
                        }
                    }
                }
            }
            Entry::Vacant(vacant) => {
                let file = Arc::new(UnlockGuard::try_lock(file, exclusive)?);

                if exclusive {
                    vacant.insert(LockState::Exclusive { extra_files: vec![] });
                } else {
                    vacant.insert(LockState::Shared(Arc::clone(&file)));
                }

                file
            }
        };

        Ok(Lock { path: p.to_owned(), file: Some(file) })
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let mut locks = LOCK_REGISTRY.lock().unwrap();
        self.file.take().unwrap();
        match locks.get_mut(&self.path).unwrap() {
            LockState::Exclusive { extra_files: _ } => {
                locks.remove(&self.path);
            }
            LockState::Shared(file) => {
                if Arc::strong_count(file) == 1 {
                    locks.remove(&self.path);
                }
            }
        }
    }
}

/// A file guard which will unlock the file when dropped.
#[derive(Debug)]
struct UnlockGuard {
    file: File,
}

impl UnlockGuard {
    fn try_lock(file: File, exclusive: bool) -> io::Result<Self> {
        let lock_type = if exclusive { libc::F_WRLCK } else { libc::F_RDLCK };

        let mut flock: libc::flock = unsafe { mem::zeroed() };
        #[cfg(not(all(target_os = "hurd", target_arch = "x86")))]
        {
            flock.l_type = lock_type as libc::c_short;
            flock.l_whence = libc::SEEK_SET as libc::c_short;
        }
        #[cfg(all(target_os = "hurd", target_arch = "x86"))]
        {
            flock.l_type = lock_type as libc::c_int;
            flock.l_whence = libc::SEEK_SET as libc::c_int;
        }
        flock.l_start = 0;
        flock.l_len = 0;

        let ret = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLK, &flock) };
        if ret == -1 { Err(io::Error::last_os_error()) } else { Ok(Self { file }) }
    }
}

impl Drop for UnlockGuard {
    fn drop(&mut self) {
        let mut flock: libc::flock = unsafe { mem::zeroed() };
        #[cfg(not(all(target_os = "hurd", target_arch = "x86")))]
        {
            flock.l_type = libc::F_UNLCK as libc::c_short;
            flock.l_whence = libc::SEEK_SET as libc::c_short;
        }
        #[cfg(all(target_os = "hurd", target_arch = "x86"))]
        {
            flock.l_type = libc::F_UNLCK as libc::c_int;
            flock.l_whence = libc::SEEK_SET as libc::c_int;
        }
        flock.l_start = 0;
        flock.l_len = 0;

        unsafe {
            libc::fcntl(self.file.as_raw_fd(), libc::F_SETLK, &flock);
        }
    }
}
