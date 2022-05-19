use crate::io;
use crate::path::Path;

#[cfg(any(
    target_os = "linux",
    target_os = "emscripten",
    target_os = "l4re",
    target_os = "android",
))]
use libc::{fdopendir, fstat64 as fstat, openat64 as openat, unlinkat};

#[cfg(not(any(
    target_os = "linux",
    target_os = "emscripten",
    target_os = "l4re",
    target_os = "android",
    all(target_os = "macos", not(target_arch = "aarch64"))
)))]
use libc::{fdopendir, fstat, openat, unlinkat};

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
use {
    libc::fstat,
    macos_compat::{fdopendir, openat, unlinkat},
};

// Dynamically resolve openat, fdopendir and unlinkat on macOS as those functions are only
// available starting with macOS 10.10. This is not necessarry for aarch64 as the first
// version supporting it was macOS 11.
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
mod macos_compat {
    use crate::sys::weak::weak;
    use libc::{c_char, c_int, DIR};

    fn get_openat_fn() -> Option<unsafe extern "C" fn(c_int, *const c_char, c_int) -> c_int> {
        weak!(fn openat(c_int, *const c_char, c_int) -> c_int);
        openat.get()
    }

    pub unsafe fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int) -> c_int {
        get_openat_fn().map(|openat| openat(dirfd, pathname, flags)).unwrap_or_else(|| {
            crate::sys::unix::os::set_errno(libc::ENOSYS);
            -1
        })
    }

    pub unsafe fn fdopendir(fd: c_int) -> *mut DIR {
        #[cfg(target_arch = "x86")]
        weak!(fn fdopendir(c_int) -> *mut DIR, "fdopendir$INODE64$UNIX2003");
        #[cfg(target_arch = "x86_64")]
        weak!(fn fdopendir(c_int) -> *mut DIR, "fdopendir$INODE64");
        fdopendir.get().map(|fdopendir| fdopendir(fd)).unwrap_or_else(|| {
            crate::sys::unix::os::set_errno(libc::ENOSYS);
            crate::ptr::null_mut()
        })
    }

    pub unsafe fn unlinkat(dirfd: c_int, pathname: *const c_char, flags: c_int) -> c_int {
        weak!(fn unlinkat(c_int, *const c_char, c_int) -> c_int);
        unlinkat.get().map(|unlinkat| unlinkat(dirfd, pathname, flags)).unwrap_or_else(|| {
            crate::sys::unix::os::set_errno(libc::ENOSYS);
            -1
        })
    }

    pub fn has_openat() -> bool {
        get_openat_fn().is_some()
    }
}

// TOCTOU-resistant implementation using openat(), unlinkat() and fdopendir()
mod remove_dir_all_xat {
    use super::{fdopendir, fstat, openat, unlinkat};
    use crate::ffi::{CStr, CString};
    use crate::io;
    use crate::mem;
    use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
    use crate::path::{Path, PathBuf};
    use crate::sys::unix::fs::{cstr, lstat, Dir, DirEntry, ReadDir};
    use crate::sys::{cvt, cvt_p, cvt_r};
    use alloc::collections::VecDeque;
    use libc::dev_t;

    #[cfg(not(any(
        target_os = "linux",
        target_os = "emscripten",
        target_os = "l4re",
        target_os = "android",
    )))]
    use libc::ino_t;

    #[cfg(any(
        target_os = "linux",
        target_os = "emscripten",
        target_os = "l4re",
        target_os = "android"
    ))]
    use libc::ino64_t as ino_t;

    fn openat_nofollow_dironly(parent_fd: Option<BorrowedFd<'_>>, p: &CStr) -> io::Result<OwnedFd> {
        let fd = cvt_r(|| unsafe {
            openat(
                parent_fd.map(|fd| fd.as_raw_fd()).unwrap_or(libc::AT_FDCWD),
                p.as_ptr(),
                libc::O_CLOEXEC | libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_DIRECTORY,
            )
        })?;
        // SAFETY: file descriptor was opened in this fn
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    #[cfg(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
    ))]
    fn is_dir(_ent: &DirEntry) -> Option<bool> {
        None
    }

    #[cfg(not(any(
        target_os = "solaris",
        target_os = "illumos",
        target_os = "haiku",
        target_os = "vxworks",
    )))]
    fn is_dir(ent: &DirEntry) -> Option<bool> {
        match ent.entry.d_type {
            libc::DT_UNKNOWN => None,
            libc::DT_DIR => Some(true),
            _ => Some(false),
        }
    }

    enum LazyReadDir<'a> {
        /// Contains the file descriptor of the directory, while it has not been opened for reading.
        /// It is only `Fd(None)` temporarily in `ensure_open()`.
        Fd(Option<OwnedFd>),

        // Contains the `ReadDir` for the directory while it is being read. The ReadDir does not contain
        // a valid `root` path, because it is not needed. It also contains the file descriptor of the
        // directory to avoid calls to dirfd(3).
        OpenReadDir(ReadDir, BorrowedFd<'a>),
    }

    impl LazyReadDir<'_> {
        fn open(path: &CStr) -> io::Result<Self> {
            let fd = openat_nofollow_dironly(None, path)?;
            Ok(LazyReadDir::Fd(Some(fd)))
        }

        fn open_subdir_or_unlink_non_dir(&self, child_name: &CStr) -> io::Result<Option<Self>> {
            let fd = match openat_nofollow_dironly(Some(self.as_fd()), child_name) {
                Ok(fd) => fd,
                Err(err) if matches!(err.raw_os_error(), Some(libc::ENOTDIR | libc::ELOOP)) => {
                    // not a directory - unlink and return
                    // (for symlinks, older Linux kernels may return ELOOP instead of ENOTDIR)
                    cvt(unsafe { unlinkat(self.as_fd().as_raw_fd(), child_name.as_ptr(), 0) })?;
                    return Ok(None);
                }
                Err(err) => return Err(err),
            };
            Ok(Some(LazyReadDir::Fd(Some(fd))))
        }

        fn get_parent(&self) -> io::Result<Self> {
            let fd = openat_nofollow_dironly(Some(self.as_fd()), unsafe {
                CStr::from_bytes_with_nul_unchecked(b"..\0")
            })?;
            Ok(LazyReadDir::Fd(Some(fd)))
        }

        fn ensure_open(&mut self) -> io::Result<()> {
            if let LazyReadDir::Fd(fd_opt) = self {
                let fd = fd_opt.take().unwrap();
                let ptr = cvt_p(unsafe { fdopendir(fd.as_raw_fd()) });
                if let Err(err) = ptr {
                    *fd_opt = Some(fd); // put the fd back
                    return Err(err);
                }
                let dirp = Dir(ptr?);
                // fd is now owned by dirp which closes it on drop, so give up ownership
                let fd = fd.into_raw_fd();
                // SAFETY: the dirp fd stays valid until self is dropped
                let fd = unsafe { BorrowedFd::borrow_raw(fd) };
                // a valid root path is not needed because we do not call any functions
                // involving the full path of the DirEntrys.
                let dummy_root = PathBuf::new();
                *self = LazyReadDir::OpenReadDir(ReadDir::new(dirp, dummy_root), fd);
            }
            Ok(())
        }
    }

    impl AsFd for LazyReadDir<'_> {
        fn as_fd(&self) -> BorrowedFd<'_> {
            match self {
                LazyReadDir::Fd(Some(fd)) => fd.as_fd(),
                LazyReadDir::Fd(None) => {
                    panic!("LazyReadDir::as_fd() called, but no fd present")
                }
                LazyReadDir::OpenReadDir(_, fd) => *fd,
            }
        }
    }

    impl Iterator for LazyReadDir<'_> {
        type Item = io::Result<DirEntry>;

        fn next(&mut self) -> Option<io::Result<DirEntry>> {
            if let Err(err) = self.ensure_open() {
                return Some(Err(err));
            }
            match self {
                LazyReadDir::OpenReadDir(rd, _) => rd.next(),
                _ => {
                    unreachable!();
                }
            }
        }
    }

    struct PathComponent {
        name: CString,
        dev: dev_t,
        ino: ino_t,
    }

    impl PathComponent {
        fn from_name_and_fd(name: &CStr, fd: BorrowedFd<'_>) -> io::Result<Self> {
            let mut stat = unsafe { mem::zeroed() };
            cvt(unsafe { fstat(fd.as_raw_fd(), &mut stat) })?;
            Ok(PathComponent { name: name.to_owned(), dev: stat.st_dev, ino: stat.st_ino })
        }

        fn verify_dev_ino(&self, fd: BorrowedFd<'_>) -> io::Result<()> {
            let mut stat = unsafe { mem::zeroed() };
            cvt(unsafe { fstat(fd.as_raw_fd(), &mut stat) })?;
            // Make sure that the reopened directory has the same inode as when we visited it descending
            // the directory tree.
            if self.dev != stat.st_dev || self.ino != stat.st_ino {
                return Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    "directory with unexpected dev/inode pair",
                ));
            }
            Ok(())
        }
    }

    fn remove_dir_all_loop(root: &Path) -> io::Result<()> {
        // VecDeque allocates space for 2^n elements if the capacity is 2^n-1.
        const MAX_OPEN_FDS: usize = 15;

        // all ancestor names and (dev, inode) pairs from the deletion root directory to the
        // parent of the directory currently being processed
        let mut path_components = Vec::new();
        // cache of the last up to MAX_OPEN_FDS ancestor ReadDirs and associated file descriptors
        let mut readdir_cache = VecDeque::with_capacity(MAX_OPEN_FDS);
        // the readdir, currently processed
        let mut current_readdir = LazyReadDir::open(&cstr(root)?)?;
        // the directory name, inode pair currently being processed
        let mut current_path_component = PathComponent::from_name_and_fd(
            unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") },
            current_readdir.as_fd(),
        )?;
        let root_parent_component = PathComponent::from_name_and_fd(
            unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") },
            current_readdir.get_parent()?.as_fd(),
        )?;
        loop {
            while let Some(child) = current_readdir.next() {
                let child = child?;
                let child_name = child.name_cstr();
                if let Some(false) = is_dir(&child) {
                    // just unlink files
                    cvt(unsafe {
                        unlinkat(current_readdir.as_fd().as_raw_fd(), child_name.as_ptr(), 0)
                    })?;
                } else {
                    if let Some(child_readdir) =
                        current_readdir.open_subdir_or_unlink_non_dir(child_name)?
                    {
                        // descend into this child directory

                        let child_path_compoment =
                            PathComponent::from_name_and_fd(child_name, child_readdir.as_fd())?;
                        path_components.push(current_path_component);
                        current_path_component = child_path_compoment;

                        // avoid growing the cache over capacity
                        if readdir_cache.len() == readdir_cache.capacity() {
                            readdir_cache.pop_front();
                        }
                        readdir_cache.push_back(current_readdir);
                        current_readdir = child_readdir;
                    }
                }
            }

            match path_components.pop() {
                Some(parent_component) => {
                    // going back up...

                    // get parent directory readdir
                    let parent_readdir = match readdir_cache.pop_back() {
                        Some(readdir) => readdir,
                        None => {
                            // cache is empty - reopen parent and grandparent fd

                            let parent_readdir = current_readdir.get_parent()?;
                            parent_component.verify_dev_ino(parent_readdir.as_fd())?;

                            // We are about to delete the now empty "child directory".
                            // To make sure the that the child directory was not moved somewhere
                            // else and that the parent just happens to have the same reused
                            // (dev, inode) pair, that we found descending, we verify the
                            // grandparent directory (dev, inode) as well.
                            let grandparent_readdir = parent_readdir.get_parent()?;
                            if let Some(grandparent_component) = path_components.last() {
                                grandparent_component
                                    .verify_dev_ino(grandparent_readdir.as_fd())?;
                                readdir_cache.push_back(grandparent_readdir);
                            } else {
                                // verify parent of the deletion root directory
                                root_parent_component
                                    .verify_dev_ino(grandparent_readdir.as_fd())?;
                            }

                            parent_readdir
                        }
                    };

                    // remove now empty directory
                    cvt(unsafe {
                        unlinkat(
                            parent_readdir.as_fd().as_raw_fd(),
                            current_path_component.name.as_ptr(),
                            libc::AT_REMOVEDIR,
                        )
                    })?;

                    current_path_component = parent_component;
                    current_readdir = parent_readdir;
                }
                None => break,
            }
        }

        // unlink deletion root directory
        cvt(unsafe { unlinkat(libc::AT_FDCWD, cstr(root)?.as_ptr(), libc::AT_REMOVEDIR) })?;
        Ok(())
    }

    pub fn remove_dir_all(p: &Path) -> io::Result<()> {
        // We cannot just call remove_dir_all_loop() here because that would not delete a passed
        // symlink. remove_dir_all_loop() does not descend into symlinks and does not delete p
        // if it is a file.
        let attr = lstat(p)?;
        if attr.file_type().is_symlink() {
            crate::fs::remove_file(p)
        } else {
            remove_dir_all_loop(p)
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub fn remove_dir_all(p: &Path) -> io::Result<()> {
    remove_dir_all_xat::remove_dir_all(p)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub fn remove_dir_all(p: &Path) -> io::Result<()> {
    if macos_compat::has_openat() {
        // openat(), unlinkat() and fdopendir() all appeared in macOS x86-64 10.10+
        remove_dir_all_xat::remove_dir_all(p)
    } else {
        // fall back to classic implementation
        crate::sys_common::fs::remove_dir_all(p)
    }
}
