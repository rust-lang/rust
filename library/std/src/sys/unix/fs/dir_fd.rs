use super::super::path::is_sep_byte;
use super::{File, OpenOptions, ReadDir};
use crate::collections::{HashSet, VecDeque};
use crate::ffi::{CStr, CString};
use crate::io;
use crate::os::unix::ffi::OsStrExt;
use crate::os::unix::fs::MetadataExt;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use crate::path::{Path, PathBuf};
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::fd::FileDesc;
use crate::sys::{cvt, cvt_r};
use crate::sys_common::FromInner;
use libc::{c_int, openat64, O_DIRECTORY, O_PATH};

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
        remove_dir_all_recursive(path)
    }
}

fn unlinkat(dirfd: BorrowedFd<'_>, path: &CStr, rmdir: bool) -> io::Result<()> {
    let flags = if rmdir { libc::AT_REMOVEDIR } else { 0 };
    cvt(unsafe { libc::unlinkat(dirfd.as_raw_fd(), path.as_ptr(), flags) })?;
    Ok(())
}

fn remove_dir_all_recursive(path: &Path) -> io::Result<()> {
    let mut dir_open_opts = OpenOptions::new();
    dir_open_opts.read(true);
    dir_open_opts.custom_flags(O_DIRECTORY);
    let root_fd = crate::fs::File::from_inner(File::open(&path, &dir_open_opts)?);
    // the current path relative to the root fd.
    let mut current_path = PathBuf::from(".");

    let mut visited_stack: Vec<(u64, u64)> = Vec::new();
    let mut visited: HashSet<(u64, u64)> = HashSet::new();

    let meta = root_fd.metadata()?;
    let root_id = (meta.ino(), meta.dev());
    visited.insert(root_id);
    visited_stack.push(root_id);

    const MAX_FDS: usize = 20;
    let mut dir_stack: VecDeque<(RawFd, ReadDir)> = VecDeque::new();

    let mut child_name_buf = Vec::new();
    let mut remove_current = false;

    'tree_walk: loop {
        if remove_current {
            let name = current_path.file_name().expect("current path empty");
            child_name_buf.clear();
            child_name_buf.extend_from_slice(name.as_bytes());
            child_name_buf.push(0);
            current_path.pop();
            let dir_id = visited_stack.pop().expect("visited stack empty");
            assert!(visited.remove(&dir_id));
        }

        let (current_dir_fd, mut current_reader) = match dir_stack.pop_back() {
            Some(dir) => dir,
            None => {
                // when ascending out of a tree deeper than MAX_FDS the ReadDirs of the parents
                // will have been dropped. reopen as needed.
                let dirfd = OwnedFd::from(crate::fs::File::from_inner(open_deep(
                    Some(root_fd.as_fd()),
                    current_path.as_path(),
                    &dir_open_opts,
                )?));
                // supply an empty pathbuf since we don't intend to use DirEntry::path()
                let rawfd = dirfd.as_raw_fd();
                (rawfd, ReadDir::from_dirfd(dirfd, PathBuf::new())?)
            }
        };

        let current_dir_fd = unsafe { BorrowedFd::borrow_raw(current_dir_fd) };

        if remove_current {
            let dir_name_c =
                unsafe { CStr::from_bytes_with_nul_unchecked(child_name_buf.as_slice()) };
            unlinkat(current_dir_fd, dir_name_c, true)?;
            remove_current = false;
        }

        while let Some(child) = current_reader.next() {
            let child = child?;

            child_name_buf.clear();
            child_name_buf.extend_from_slice(child.file_name_os_str().as_bytes());
            child_name_buf.push(0);

            let child_name_c =
                unsafe { CStr::from_bytes_with_nul_unchecked(child_name_buf.as_slice()) };

            if child.file_type()?.is_dir() {
                current_path.push(child.file_name_os_str());

                let dir = crate::fs::File::from_inner(File::open_c(
                    Some(current_dir_fd),
                    child_name_c,
                    &dir_open_opts,
                )?);

                dir_stack.push_back((current_dir_fd.as_raw_fd(), current_reader));

                // since we're traversing the directory tree via openat we have to do
                // cycle-detection ourselves
                let meta = dir.metadata()?;
                let dir_id = (meta.ino(), meta.dev());
                if visited.contains(&dir_id) {
                    return Err(io::Error::from_raw_os_error(libc::ELOOP));
                }
                visited_stack.push(dir_id);
                visited.insert(dir_id);

                let dirfd = OwnedFd::from(dir);
                let raw_fd = dirfd.as_raw_fd();
                let reader = ReadDir::from_dirfd(dirfd, PathBuf::new())?;
                dir_stack.push_back((raw_fd, reader));

                if dir_stack.len() > MAX_FDS {
                    dir_stack.pop_front();
                }

                continue 'tree_walk;
            }

            unlinkat(current_dir_fd, child_name_c, false)?;
        }

        // reached end of directory
        if current_path.as_os_str().len() == 1 {
            break;
        }

        remove_current = true;
    }

    // There's no dirfd to reuse here since unlinking the starting point requires unlinking relative to
    // the parent directory. '..' also cannot be used because that is vulnerable to the directory
    // being moved. And we couldn't have started with the parent directory either because the starting
    // point may be '/' or '.' which can't be removed themselves but we still should be able to clean
    // their descendants.
    super::rmdir(path)
}
