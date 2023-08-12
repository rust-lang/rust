use super::super::path::is_sep_byte;
use super::{File, OpenOptions, ReadDir};
use crate::ffi::{CStr, CString};
use crate::io;
use crate::os::unix::ffi::OsStrExt;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use crate::path::{Path, PathBuf};
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::fd::FileDesc;
use crate::sys::{cvt, cvt_r};
use libc::{c_int, openat64, O_PATH};

impl ReadDir {
    pub fn from_dirfd(dirfd: OwnedFd, root: PathBuf) -> io::Result<ReadDir> {
        let dir_pointer = super::cvt_p(unsafe { libc::fdopendir(dirfd.as_raw_fd()) })?;
        // fdopendir takes ownership on success
        crate::mem::forget(dirfd);
        Ok(Self::from_dirp(dir_pointer, root))
    }
}

impl File {
    pub fn open_c(
        dirfd: Option<BorrowedFd<'_>>,
        path: &CStr,
        opts: &OpenOptions,
    ) -> io::Result<File> {
        let flags = libc::O_CLOEXEC
            | opts.get_access_mode()?
            | opts.get_creation_mode()?
            | (opts.custom_flags as c_int & !libc::O_ACCMODE);

        let dirfd = match dirfd {
            None => libc::AT_FDCWD,
            Some(dirfd) => dirfd.as_raw_fd(),
        };
        let fd = cvt_r(|| unsafe {
            openat64(
                dirfd,
                path.as_ptr(),
                flags,
                // see previous comment why this cast is necessary
                opts.mode as c_int,
            )
        })?;

        Ok(File(unsafe { FileDesc::from_raw_fd(fd) }))
    }
}

pub fn open_deep(
    at_path: Option<BorrowedFd<'_>>,
    path: &Path,
    opts: &OpenOptions,
) -> io::Result<File> {
    const MAX_SLICE: usize = (libc::PATH_MAX - 1) as usize;

    enum AtPath<'a> {
        None,
        Borrowed(BorrowedFd<'a>),
        File(File),
    }

    impl<'a> AtPath<'a> {
        fn as_fd(&'a self) -> Option<BorrowedFd<'a>> {
            match self {
                AtPath::Borrowed(borrowed) => Some(*borrowed),
                AtPath::File(ref file) => Some(file.as_fd()),
                AtPath::None => None,
            }
        }
    }

    let mut raw_path = path.as_os_str().as_bytes();
    let mut at_path = match at_path {
        Some(borrowed) => AtPath::Borrowed(borrowed),
        None => AtPath::None,
    };

    let mut dir_flags = OpenOptions::new();
    dir_flags.read(true);
    dir_flags.custom_flags(O_PATH);

    while raw_path.len() > MAX_SLICE {
        let sep_idx = match raw_path.iter().take(MAX_SLICE).rposition(|&byte| is_sep_byte(byte)) {
            Some(idx) => idx,
            _ => return Err(io::Error::from_raw_os_error(libc::ENAMETOOLONG)),
        };

        let (left, right) = raw_path.split_at(sep_idx + 1);
        raw_path = right;

        let to_open = CString::new(left)?;
        let dirfd = at_path.as_fd();

        at_path = AtPath::File(File::open_c(dirfd, &to_open, &dir_flags)?);
    }

    let to_open = CString::new(raw_path)?;
    let dirfd = at_path.as_fd();

    File::open_c(dirfd, &to_open, opts)
}

pub fn long_filename_fallback<T>(
    result: io::Result<T>,
    path: &Path,
    mut fallback: impl FnMut(File, &CStr) -> io::Result<T>,
) -> io::Result<T> {
    use crate::io::ErrorKind;
    match result {
        ok @ Ok(_) => ok,
        Err(e) if e.kind() == ErrorKind::InvalidFilename => {
            if let Some(parent) = path.parent() {
                let mut options = OpenOptions::new();
                options.read(true);
                options.custom_flags(libc::O_PATH);
                let dirfd = open_deep(None, parent, &options)?;
                let file_name = path.file_name().unwrap();
                return run_path_with_cstr(file_name, |file_name| fallback(dirfd, file_name));
            }

            Err(e)
        }
        Err(e) => Err(e),
    }
}

pub fn remove_dir_all(path: &Path) -> io::Result<()> {
    let filetype = crate::fs::symlink_metadata(path)?.file_type();
    if filetype.is_symlink() {
        crate::fs::remove_file(path)
    } else {
        rmdir::remove_dir_all_iter(path)
    }
}

fn unlinkat(dirfd: BorrowedFd<'_>, path: &CStr, rmdir: bool) -> io::Result<()> {
    let flags = if rmdir { libc::AT_REMOVEDIR } else { 0 };
    cvt(unsafe { libc::unlinkat(dirfd.as_raw_fd(), path.as_ptr(), flags) })?;
    Ok(())
}

mod rmdir {
    use crate::ffi::CStr;

    use crate::collections::HashSet;
    use crate::ffi::OsStr;
    use crate::fs::Metadata;
    use crate::io;
    use crate::io::Result;
    use crate::os::unix::ffi::OsStrExt;
    use crate::os::unix::fs::MetadataExt;
    use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
    use crate::path::{Path, PathBuf};
    use crate::sys::fs::{File, OpenOptions, ReadDir};
    use crate::sys_common::FromInner;
    use libc::{O_DIRECTORY, O_NOFOLLOW, O_PATH};

    use super::{open_deep, unlinkat};

    #[derive(PartialEq, Eq, Hash, Copy, Clone)]
    struct DirId(u64, u64);

    impl From<Metadata> for DirId {
        fn from(meta: Metadata) -> Self {
            DirId(meta.dev(), meta.ino())
        }
    }

    struct DirStack {
        /// dirid specified the device id + inode number
        /// for consistency checking.
        /// The option carries the readdir and the file descriptor
        /// from which it has been constructed.
        /// Dropping the ReadDir invalidates the fd.
        /// The first entry is the root directory whose
        /// contents we want to delete.
        dirs: Vec<(DirId, Option<(RawFd, ReadDir)>)>,
        /// Each name component in the pathbuf represents
        /// one entry in the `dirs` vec.
        names: PathBuf,
        /// Used for loop detection
        visited: HashSet<DirId>,
        /// temporary buffer to construct a CStr for syscalls.
        child_name_buffer: Vec<u8>,
    }

    fn dir_open_options() -> OpenOptions {
        let mut opts = OpenOptions::new();
        opts.read(true);
        opts.custom_flags(O_DIRECTORY | O_NOFOLLOW);
        opts
    }

    const MAX_FDS: usize = 20;

    impl DirStack {
        fn new(root_fd: crate::fs::File) -> Result<Self> {
            let mut stack = DirStack {
                dirs: Vec::new(),
                names: PathBuf::from("./"),
                visited: HashSet::new(),
                child_name_buffer: Vec::new(),
            };

            let meta = root_fd.metadata()?;
            let root_id = meta.into();

            let raw_fd = root_fd.as_raw_fd();
            let dirfd = OwnedFd::from(root_fd);
            let reader = ReadDir::from_dirfd(dirfd, PathBuf::new())?;

            stack.visited.insert(root_id);
            stack.dirs.push((root_id, Some((raw_fd, reader))));

            stack.check_invariants(true);

            Ok(stack)
        }

        fn ensure_open(&mut self) -> Result<()> {
            self.check_invariants(false);

            // Don't refill ancestors until the last one has been popped off.
            // Doing it in batches reduces traversal/checking costs.
            if self.dirs.last().unwrap().1.is_none() {
                let start = self.dirs.len().saturating_sub(MAX_FDS).max(1);
                let end = self.dirs.len();
                for idx in start..end {
                    if self.dirs[idx].1.is_none() {
                        let nearest_ancestor_idx = self.dirs[..idx]
                            .iter()
                            .rposition(|(_, fd)| fd.is_some())
                            .expect("some open ancestor");
                        let (ancestor_id, ancestor_fd) = &self.dirs[nearest_ancestor_idx];
                        let ancestor_id = *ancestor_id;
                        let mut path_components = self.names.components();
                        path_components
                            .advance_by(nearest_ancestor_idx + 1)
                            .expect("advanced too far");
                        path_components
                            .advance_back_by(self.dirs.len() - idx - 1)
                            .expect("advanced_back too far");
                        let sub_path = path_components.as_path();
                        let depth = idx - nearest_ancestor_idx;

                        debug_assert_eq!(sub_path.components().count(), depth);

                        let ancestor_fd =
                            unsafe { BorrowedFd::borrow_raw(ancestor_fd.as_ref().unwrap().0) };

                        // use open_deep since the relative path can be arbitrarily long
                        let dir = crate::fs::File::from_inner(open_deep(
                            Some(ancestor_fd),
                            sub_path,
                            &dir_open_options(),
                        )?);
                        let meta = dir.metadata()?;
                        let dir_id = DirId(meta.ino(), meta.dev());

                        let (expected_id, entry) = &mut self.dirs[idx];

                        // Security check to prevent TOCTOU attacks when re-traversing the hierarchy.
                        // `open_deep` follows symlinks, so verify that we arrive at the same spot
                        // we have been at before.
                        // But this leaves the possibility of a symlink + inode recycling race...
                        if dir_id != *expected_id {
                            return Err(io::const_io_error!(
                                io::ErrorKind::Uncategorized,
                                "directory with unexpected dev/ino",
                            ));
                        }
                        // ... so we make extra sure that the directory is a descendant by going back up via `../..`
                        // to the nearest still-open ancestor. The ancestor being open prevents inode recycling.
                        // This could be avoided if open_deep used used openat2(..., O_BENEATH)
                        // but that's only available in linux kernels >= 5.6
                        if depth > 1 {
                            let mut buf = PathBuf::new();
                            for _ in 0..depth {
                                buf.push("..")
                            }
                            let mut opts = OpenOptions::new();
                            opts.read(true);
                            // open as path handle since we only want to stat it
                            opts.custom_flags(O_DIRECTORY | O_PATH);

                            let ancestor_o_path = crate::fs::File::from_inner(open_deep(
                                Some(dir.as_fd()),
                                &buf,
                                &opts,
                            )?);
                            let meta = ancestor_o_path.metadata()?;

                            if DirId(meta.dev(), meta.ino()) != ancestor_id {
                                return Err(io::const_io_error!(
                                    io::ErrorKind::Uncategorized,
                                    "unexpected ancestor dir dev/ino",
                                ));
                            }
                        }

                        let dirfd = OwnedFd::from(dir);
                        let rawfd = dirfd.as_raw_fd();

                        // supply an empty pathbuf since we don't intend to use DirEntry::path()
                        *entry = Some((rawfd, ReadDir::from_dirfd(dirfd, PathBuf::new())?));
                    }
                }
            }

            self.check_invariants(true);

            Ok(())
        }

        fn sparsify(&mut self) {
            self.check_invariants(true);

            // Keep the root fd and MAX_FDS tail entries open, close the rest.
            //
            // We could do something more clever here such as keeping a log(n)
            // intermediate hops with expoenntial spacing to amortize reopening
            // costs in very very deep directory trees.
            for entry in self.dirs.iter_mut().skip(1).rev().skip(MAX_FDS) {
                if entry.1.is_some() {
                    entry.1 = None;
                } else {
                    break;
                }
            }
        }

        fn check_invariants(&self, require_current_open: bool) {
            debug_assert_eq!(self.names.components().count(), self.dirs.len());
            debug_assert!(self.dirs.len() > 0);
            debug_assert!(self.dirs[0].1.is_some());
            if require_current_open {
                debug_assert!(self.dirs.last().unwrap().1.is_some());
            }
        }

        fn pop(&mut self) -> Result<()> {
            self.check_invariants(true);
            debug_assert!(self.dirs.len() > 1);

            let (id, fd) = self.dirs.pop().unwrap();
            drop(fd);

            let name = self.names.file_name().expect("path should not be empty");
            let mut buf = crate::mem::take(&mut self.child_name_buffer);
            buf.clear();
            buf.extend_from_slice(name.as_bytes());
            buf.push(0);

            self.names.pop();
            self.visited.remove(&id);

            self.ensure_open()?;

            let parent = self.dirs.last().expect("at least the root should still be open");

            let current_dir_fd = unsafe { BorrowedFd::borrow_raw(parent.1.as_ref().unwrap().0) };

            let dir_name_c = unsafe { CStr::from_bytes_with_nul_unchecked(buf.as_slice()) };
            unlinkat(current_dir_fd, dir_name_c, true)?;
            self.child_name_buffer = buf;

            Ok(())
        }

        fn push(&mut self, dir_name: &OsStr) -> Result<()> {
            debug_assert!(dir_name.len() > 0);
            self.check_invariants(true);

            let parent_fd = self.dirs.last().unwrap().1.as_ref().unwrap().0;

            let buf = &mut self.child_name_buffer;
            buf.clear();
            buf.extend_from_slice(dir_name.as_bytes());
            buf.push(0);
            let dir_name_c = unsafe { CStr::from_bytes_with_nul_unchecked(buf.as_slice()) };

            let dir = crate::fs::File::from_inner(File::open_c(
                Some(unsafe { BorrowedFd::borrow_raw(parent_fd) }),
                dir_name_c,
                &dir_open_options(),
            )?);
            let meta = dir.metadata()?;
            let dir_id = DirId(meta.ino(), meta.dev());

            if !self.visited.insert(dir_id) {
                return Err(io::Error::from_raw_os_error(libc::ELOOP));
            }

            let dirfd = OwnedFd::from(dir);
            let rawfd = dirfd.as_raw_fd();

            let entry = (dir_id, Some((rawfd, ReadDir::from_dirfd(dirfd, PathBuf::new())?)));

            self.dirs.push(entry);
            self.names.push(dir_name);

            self.sparsify();

            Ok(())
        }

        fn walk_tree(&mut self) -> Result<()> {
            loop {
                self.check_invariants(true);
                let (fd, current_dir) = self
                    .dirs
                    .last_mut()
                    .expect("at least the root should be open")
                    .1
                    .as_mut()
                    .expect("the last entry shouldn't have its FD closed");

                match current_dir.next() {
                    Some(child) => {
                        let child = child?;
                        let child_name = child.file_name();

                        if child.file_type()?.is_dir() {
                            self.push(&child_name)?;
                            continue;
                        }

                        let buf = &mut self.child_name_buffer;
                        buf.clear();
                        buf.extend_from_slice(child_name.as_os_str().as_bytes());
                        buf.push(0);
                        let child_name_c =
                            unsafe { CStr::from_bytes_with_nul_unchecked(buf.as_slice()) };

                        let fd = unsafe { BorrowedFd::borrow_raw(*fd) };
                        unlinkat(fd, child_name_c, false)?;
                    }
                    None if self.dirs.len() > 1 => {
                        self.pop()?;
                    }
                    None => break,
                }
            }

            Ok(())
        }
    }

    pub(super) fn remove_dir_all_iter(path: &Path) -> Result<()> {
        let root_fd = crate::fs::File::from_inner(File::open(&path, &dir_open_options())?);

        let mut stack = DirStack::new(root_fd)?;
        stack.walk_tree()?;

        // There's no dirfd to reuse here since unlinking the starting point requires unlinking relative to
        // the parent directory. '..' also cannot be used because that is vulnerable to the directory
        // being moved. And we couldn't have started with the parent directory either because the starting
        // point may be '/' or '.' which can't be removed themselves but we still should be able to clean
        // their descendants.
        crate::sys::fs::rmdir(path)
    }
}
