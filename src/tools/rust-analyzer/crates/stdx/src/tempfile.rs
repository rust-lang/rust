//! A temporary named file that will be deleted on drop, and on operating systems that support that,
//! also when the process exits (including being killed).

use std::{
    fs::File,
    io,
    path::{Path, PathBuf},
};

pub struct NamedTempFile {
    _file: Option<File>,
    path: PathBuf,
    delete_on_drop: bool,
}

impl NamedTempFile {
    pub fn new(prefix: &str) -> io::Result<NamedTempFile> {
        imp::create(prefix)
    }

    /// Creates a new `NamedTempFile` that is a copy of an existing file.
    pub fn new_from_existing(prefix: &str, existing: &Path) -> io::Result<NamedTempFile> {
        let result = NamedTempFile::new(prefix)?;
        std::fs::copy(existing, &result.path)?;
        Ok(result)
    }

    /// Creates a `NamedTempFile` from a path, without deleting it on drop.
    #[inline]
    pub fn from_path(path: PathBuf) -> NamedTempFile {
        NamedTempFile { _file: None, path, delete_on_drop: false }
    }

    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for NamedTempFile {
    fn drop(&mut self) {
        if self.delete_on_drop && std::fs::remove_file(&self.path).is_err() {
            tracing::info!("cannot remove temporary file {}", self.path.display());
        }
    }
}

mod general_imp {
    use std::{
        fs::{File, OpenOptions},
        io::{self, ErrorKind},
        path::PathBuf,
        sync::atomic::{AtomicU32, Ordering},
    };

    static INTERNAL_COUNTER: AtomicU32 = AtomicU32::new(0);

    pub(super) fn create(
        prefix: &str,
        mut options_callback: impl FnMut(&mut OpenOptions),
    ) -> io::Result<(File, PathBuf)> {
        let temp_dir = std::env::temp_dir().canonicalize()?;
        let pid = std::process::id();
        loop {
            let path = temp_dir.join(format!(
                "{prefix}{pid:x}-{:x}",
                INTERNAL_COUNTER.fetch_add(1, Ordering::AcqRel),
            ));
            let mut open_options = OpenOptions::new();
            open_options.create_new(true);
            options_callback(&mut open_options);
            match open_options.open(&path) {
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
                Err(e) => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!("error creating directory {path:?}: {e}"),
                    ));
                }
                Ok(file) => {
                    return Ok((file, path));
                }
            }
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
))]
mod imp {
    use std::{
        ffi::CString,
        io,
        os::{
            fd::{AsRawFd, RawFd},
            unix::ffi::OsStrExt,
        },
    };

    use super::*;

    #[cfg(target_os = "linux")]
    fn path_after_unlink(fd: RawFd) -> PathBuf {
        PathBuf::from(format!("/proc/self/fd/{fd}"))
    }

    #[cfg(any(target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
    fn path_after_unlink(fd: RawFd) -> PathBuf {
        PathBuf::from(format!("/dev/fd/{fd}"))
    }

    pub(super) fn create(prefix: &str) -> io::Result<NamedTempFile> {
        let (file, mut path) = general_imp::create(prefix, |_| {})?;
        let mut delete_on_drop = true;
        if let Ok(original_path) = CString::new(path.as_os_str().as_bytes()) {
            // Unlinking the file will *not* remove it per the POSIX specification since it is open.
            // We cannot use `std::fs::remove_file()`, since, while currently using `unlink()`, it does
            // not guarantee it will use it.
            if unsafe { libc::unlink(original_path.as_ptr()) } == 0 {
                path = path_after_unlink(file.as_raw_fd());
                delete_on_drop = false;
            }
        }
        Ok(NamedTempFile { _file: Some(file), path, delete_on_drop })
    }
}

#[cfg(windows)]
mod imp {
    use std::os::windows::fs::OpenOptionsExt;

    use super::*;

    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x04000000;

    pub(super) fn create(prefix: &str) -> io::Result<NamedTempFile> {
        let (file, path) = general_imp::create(prefix, |options| {
            options.attributes(FILE_ATTRIBUTE_TEMPORARY);
            options.custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
        })?;
        Ok(NamedTempFile { _file: Some(file), path, delete_on_drop: false })
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    windows,
)))]
mod imp {
    use super::*;

    pub(super) fn create(prefix: &str) -> io::Result<NamedTempFile> {
        let (file, path) = general_imp::create(prefix, |_| {})?;
        Ok(NamedTempFile { _file: Some(file), path, delete_on_drop: true })
    }
}
