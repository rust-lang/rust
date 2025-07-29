//! File and file system access

use std::borrow::Cow;
use std::fs::{
    DirBuilder, File, FileType, OpenOptions, ReadDir, TryLockError, read_dir, remove_dir,
    remove_file, rename,
};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rustc_abi::Size;
use rustc_data_structures::fx::FxHashMap;

use self::shims::time::system_time_to_duration;
use crate::shims::files::FileHandle;
use crate::shims::os_str::bytes_to_os_str;
use crate::shims::sig::check_min_vararg_count;
use crate::shims::unix::fd::{FlockOp, UnixFileDescription};
use crate::*;

impl UnixFileDescription for FileHandle {
    fn pread<'tcx>(
        &self,
        communicate_allowed: bool,
        offset: u64,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        let mut bytes = vec![0; len];
        // Emulates pread using seek + read + seek to restore cursor position.
        // Correctness of this emulation relies on sequential nature of Miri execution.
        // The closure is used to emulate `try` block, since we "bubble" `io::Error` using `?`.
        let file = &mut &self.file;
        let mut f = || {
            let cursor_pos = file.stream_position()?;
            file.seek(SeekFrom::Start(offset))?;
            let res = file.read(&mut bytes);
            // Attempt to restore cursor position even if the read has failed
            file.seek(SeekFrom::Start(cursor_pos))
                .expect("failed to restore file position, this shouldn't be possible");
            res
        };
        let result = match f() {
            Ok(read_size) => {
                // If reading to `bytes` did not fail, we write those bytes to the buffer.
                // Crucially, if fewer than `bytes.len()` bytes were read, only write
                // that much into the output buffer!
                ecx.write_bytes_ptr(ptr, bytes[..read_size].iter().copied())?;
                Ok(read_size)
            }
            Err(e) => Err(IoError::HostError(e)),
        };
        finish.call(ecx, result)
    }

    fn pwrite<'tcx>(
        &self,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        offset: u64,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        // Emulates pwrite using seek + write + seek to restore cursor position.
        // Correctness of this emulation relies on sequential nature of Miri execution.
        // The closure is used to emulate `try` block, since we "bubble" `io::Error` using `?`.
        let file = &mut &self.file;
        let bytes = ecx.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        let mut f = || {
            let cursor_pos = file.stream_position()?;
            file.seek(SeekFrom::Start(offset))?;
            let res = file.write(bytes);
            // Attempt to restore cursor position even if the write has failed
            file.seek(SeekFrom::Start(cursor_pos))
                .expect("failed to restore file position, this shouldn't be possible");
            res
        };
        let result = f();
        finish.call(ecx, result.map_err(IoError::HostError))
    }

    fn flock<'tcx>(
        &self,
        communicate_allowed: bool,
        op: FlockOp,
    ) -> InterpResult<'tcx, io::Result<()>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");

        use FlockOp::*;
        // We must not block the interpreter loop, so we always `try_lock`.
        let (res, nonblocking) = match op {
            SharedLock { nonblocking } => (self.file.try_lock_shared(), nonblocking),
            ExclusiveLock { nonblocking } => (self.file.try_lock(), nonblocking),
            Unlock => {
                return interp_ok(self.file.unlock());
            }
        };

        match res {
            Ok(()) => interp_ok(Ok(())),
            Err(TryLockError::Error(err)) => interp_ok(Err(err)),
            Err(TryLockError::WouldBlock) =>
                if nonblocking {
                    interp_ok(Err(ErrorKind::WouldBlock.into()))
                } else {
                    throw_unsup_format!("blocking `flock` is not currently supported");
                },
        }
    }
}

impl<'tcx> EvalContextExtPrivate<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPrivate<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn macos_fbsd_solarish_write_stat_buf(
        &mut self,
        metadata: FileMetadata,
        buf_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let (access_sec, access_nsec) = metadata.accessed.unwrap_or((0, 0));
        let (created_sec, created_nsec) = metadata.created.unwrap_or((0, 0));
        let (modified_sec, modified_nsec) = metadata.modified.unwrap_or((0, 0));
        let mode = metadata.mode.to_uint(this.libc_ty_layout("mode_t").size)?;

        let buf = this.deref_pointer_as(buf_op, this.libc_ty_layout("stat"))?;
        this.write_int_fields_named(
            &[
                ("st_dev", metadata.dev.into()),
                ("st_mode", mode.try_into().unwrap()),
                ("st_nlink", 0),
                ("st_ino", 0),
                ("st_uid", metadata.uid.into()),
                ("st_gid", metadata.gid.into()),
                ("st_rdev", 0),
                ("st_atime", access_sec.into()),
                ("st_mtime", modified_sec.into()),
                ("st_ctime", 0),
                ("st_size", metadata.size.into()),
                ("st_blocks", 0),
                ("st_blksize", 0),
            ],
            &buf,
        )?;

        if matches!(&*this.tcx.sess.target.os, "macos" | "freebsd") {
            this.write_int_fields_named(
                &[
                    ("st_atime_nsec", access_nsec.into()),
                    ("st_mtime_nsec", modified_nsec.into()),
                    ("st_ctime_nsec", 0),
                    ("st_birthtime", created_sec.into()),
                    ("st_birthtime_nsec", created_nsec.into()),
                    ("st_flags", 0),
                    ("st_gen", 0),
                ],
                &buf,
            )?;
        }

        if matches!(&*this.tcx.sess.target.os, "solaris" | "illumos") {
            let st_fstype = this.project_field_named(&buf, "st_fstype")?;
            // This is an array; write 0 into first element so that it encodes the empty string.
            this.write_int(0, &this.project_index(&st_fstype, 0)?)?;
        }

        interp_ok(0)
    }

    fn file_type_to_d_type(
        &mut self,
        file_type: std::io::Result<FileType>,
    ) -> InterpResult<'tcx, i32> {
        #[cfg(unix)]
        use std::os::unix::fs::FileTypeExt;

        let this = self.eval_context_mut();
        match file_type {
            Ok(file_type) => {
                match () {
                    _ if file_type.is_dir() => interp_ok(this.eval_libc("DT_DIR").to_u8()?.into()),
                    _ if file_type.is_file() => interp_ok(this.eval_libc("DT_REG").to_u8()?.into()),
                    _ if file_type.is_symlink() =>
                        interp_ok(this.eval_libc("DT_LNK").to_u8()?.into()),
                    // Certain file types are only supported when the host is a Unix system.
                    #[cfg(unix)]
                    _ if file_type.is_block_device() =>
                        interp_ok(this.eval_libc("DT_BLK").to_u8()?.into()),
                    #[cfg(unix)]
                    _ if file_type.is_char_device() =>
                        interp_ok(this.eval_libc("DT_CHR").to_u8()?.into()),
                    #[cfg(unix)]
                    _ if file_type.is_fifo() =>
                        interp_ok(this.eval_libc("DT_FIFO").to_u8()?.into()),
                    #[cfg(unix)]
                    _ if file_type.is_socket() =>
                        interp_ok(this.eval_libc("DT_SOCK").to_u8()?.into()),
                    // Fallback
                    _ => interp_ok(this.eval_libc("DT_UNKNOWN").to_u8()?.into()),
                }
            }
            Err(_) => {
                // Fallback on error
                interp_ok(this.eval_libc("DT_UNKNOWN").to_u8()?.into())
            }
        }
    }
}

/// An open directory, tracked by DirHandler.
#[derive(Debug)]
struct OpenDir {
    /// The directory reader on the host.
    read_dir: ReadDir,
    /// The most recent entry returned by readdir().
    /// Will be freed by the next call.
    entry: Option<Pointer>,
}

impl OpenDir {
    fn new(read_dir: ReadDir) -> Self {
        Self { read_dir, entry: None }
    }
}

/// The table of open directories.
/// Curiously, Unix/POSIX does not unify this into the "file descriptor" concept... everything
/// is a file, except a directory is not?
#[derive(Debug)]
pub struct DirTable {
    /// Directory iterators used to emulate libc "directory streams", as used in opendir, readdir,
    /// and closedir.
    ///
    /// When opendir is called, a directory iterator is created on the host for the target
    /// directory, and an entry is stored in this hash map, indexed by an ID which represents
    /// the directory stream. When readdir is called, the directory stream ID is used to look up
    /// the corresponding ReadDir iterator from this map, and information from the next
    /// directory entry is returned. When closedir is called, the ReadDir iterator is removed from
    /// the map.
    streams: FxHashMap<u64, OpenDir>,
    /// ID number to be used by the next call to opendir
    next_id: u64,
}

impl DirTable {
    #[expect(clippy::arithmetic_side_effects)]
    fn insert_new(&mut self, read_dir: ReadDir) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.streams.try_insert(id, OpenDir::new(read_dir)).unwrap();
        id
    }
}

impl Default for DirTable {
    fn default() -> DirTable {
        DirTable {
            streams: FxHashMap::default(),
            // Skip 0 as an ID, because it looks like a null pointer to libc
            next_id: 1,
        }
    }
}

impl VisitProvenance for DirTable {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let DirTable { streams, next_id: _ } = self;

        for dir in streams.values() {
            dir.entry.visit_provenance(visit);
        }
    }
}

fn maybe_sync_file(
    file: &File,
    writable: bool,
    operation: fn(&File) -> std::io::Result<()>,
) -> std::io::Result<i32> {
    if !writable && cfg!(windows) {
        // sync_all() and sync_data() will return an error on Windows hosts if the file is not opened
        // for writing. (FlushFileBuffers requires that the file handle have the
        // GENERIC_WRITE right)
        Ok(0i32)
    } else {
        let result = operation(file);
        result.map(|_| 0i32)
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn open(
        &mut self,
        path_raw: &OpTy<'tcx>,
        flag: &OpTy<'tcx>,
        varargs: &[OpTy<'tcx>],
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let path_raw = this.read_pointer(path_raw)?;
        let path = this.read_path_from_c_str(path_raw)?;
        let flag = this.read_scalar(flag)?.to_i32()?;

        let mut options = OpenOptions::new();

        let o_rdonly = this.eval_libc_i32("O_RDONLY");
        let o_wronly = this.eval_libc_i32("O_WRONLY");
        let o_rdwr = this.eval_libc_i32("O_RDWR");
        // The first two bits of the flag correspond to the access mode in linux, macOS and
        // windows. We need to check that in fact the access mode flags for the current target
        // only use these two bits, otherwise we are in an unsupported target and should error.
        if (o_rdonly | o_wronly | o_rdwr) & !0b11 != 0 {
            throw_unsup_format!("access mode flags on this target are unsupported");
        }
        let mut writable = true;

        // Now we check the access mode
        let access_mode = flag & 0b11;

        if access_mode == o_rdonly {
            writable = false;
            options.read(true);
        } else if access_mode == o_wronly {
            options.write(true);
        } else if access_mode == o_rdwr {
            options.read(true).write(true);
        } else {
            throw_unsup_format!("unsupported access mode {:#x}", access_mode);
        }
        // We need to check that there aren't unsupported options in `flag`. For this we try to
        // reproduce the content of `flag` in the `mirror` variable using only the supported
        // options.
        let mut mirror = access_mode;

        let o_append = this.eval_libc_i32("O_APPEND");
        if flag & o_append == o_append {
            options.append(true);
            mirror |= o_append;
        }
        let o_trunc = this.eval_libc_i32("O_TRUNC");
        if flag & o_trunc == o_trunc {
            options.truncate(true);
            mirror |= o_trunc;
        }
        let o_creat = this.eval_libc_i32("O_CREAT");
        if flag & o_creat == o_creat {
            // Get the mode.  On macOS, the argument type `mode_t` is actually `u16`, but
            // C integer promotion rules mean that on the ABI level, it gets passed as `u32`
            // (see https://github.com/rust-lang/rust/issues/71915).
            let [mode] = check_min_vararg_count("open(pathname, O_CREAT, ...)", varargs)?;
            let mode = this.read_scalar(mode)?.to_u32()?;

            #[cfg(unix)]
            {
                // Support all modes on UNIX host
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(mode);
            }
            #[cfg(not(unix))]
            {
                // Only support default mode for non-UNIX (i.e. Windows) host
                if mode != 0o666 {
                    throw_unsup_format!(
                        "non-default mode 0o{:o} is not supported on non-Unix hosts",
                        mode
                    );
                }
            }

            mirror |= o_creat;

            let o_excl = this.eval_libc_i32("O_EXCL");
            if flag & o_excl == o_excl {
                mirror |= o_excl;
                options.create_new(true);
            } else {
                options.create(true);
            }
        }
        let o_cloexec = this.eval_libc_i32("O_CLOEXEC");
        if flag & o_cloexec == o_cloexec {
            // We do not need to do anything for this flag because `std` already sets it.
            // (Technically we do not support *not* setting this flag, but we ignore that.)
            mirror |= o_cloexec;
        }
        if this.tcx.sess.target.os == "linux" {
            let o_tmpfile = this.eval_libc_i32("O_TMPFILE");
            if flag & o_tmpfile == o_tmpfile {
                // if the flag contains `O_TMPFILE` then we return a graceful error
                return this.set_last_error_and_return_i32(LibcError("EOPNOTSUPP"));
            }
        }

        let o_nofollow = this.eval_libc_i32("O_NOFOLLOW");
        if flag & o_nofollow == o_nofollow {
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.custom_flags(libc::O_NOFOLLOW);
            }
            // Strictly speaking, this emulation is not equivalent to the O_NOFOLLOW flag behavior:
            // the path could change between us checking it here and the later call to `open`.
            // But it's good enough for Miri purposes.
            #[cfg(not(unix))]
            {
                // O_NOFOLLOW only fails when the trailing component is a symlink;
                // the entire rest of the path can still contain symlinks.
                if path.is_symlink() {
                    return this.set_last_error_and_return_i32(LibcError("ELOOP"));
                }
            }
            mirror |= o_nofollow;
        }

        // If `flag` is not equal to `mirror`, there is an unsupported option enabled in `flag`,
        // then we throw an error.
        if flag != mirror {
            throw_unsup_format!("unsupported flags {:#x}", flag & !mirror);
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`open`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let fd = options
            .open(path)
            .map(|file| this.machine.fds.insert_new(FileHandle { file, writable }));

        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(fd)?))
    }

    fn lseek64(
        &mut self,
        fd_num: i32,
        offset: i128,
        whence: i32,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescription` trait.

        let seek_from = if whence == this.eval_libc_i32("SEEK_SET") {
            if offset < 0 {
                // Negative offsets return `EINVAL`.
                return this.set_last_error_and_return(LibcError("EINVAL"), dest);
            } else {
                SeekFrom::Start(u64::try_from(offset).unwrap())
            }
        } else if whence == this.eval_libc_i32("SEEK_CUR") {
            SeekFrom::Current(i64::try_from(offset).unwrap())
        } else if whence == this.eval_libc_i32("SEEK_END") {
            SeekFrom::End(i64::try_from(offset).unwrap())
        } else {
            return this.set_last_error_and_return(LibcError("EINVAL"), dest);
        };

        let communicate = this.machine.communicate();

        let Some(fd) = this.machine.fds.get(fd_num) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };
        let result = fd.seek(communicate, seek_from)?.map(|offset| i64::try_from(offset).unwrap());
        drop(fd);

        let result = this.try_unwrap_io_result(result)?;
        this.write_int(result, dest)?;
        interp_ok(())
    }

    fn unlink(&mut self, path_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`unlink`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let result = remove_file(path).map(|_| 0);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn symlink(
        &mut self,
        target_op: &OpTy<'tcx>,
        linkpath_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        #[cfg(unix)]
        fn create_link(src: &Path, dst: &Path) -> std::io::Result<()> {
            std::os::unix::fs::symlink(src, dst)
        }

        #[cfg(windows)]
        fn create_link(src: &Path, dst: &Path) -> std::io::Result<()> {
            use std::os::windows::fs;
            if src.is_dir() { fs::symlink_dir(src, dst) } else { fs::symlink_file(src, dst) }
        }

        let this = self.eval_context_mut();
        let target = this.read_path_from_c_str(this.read_pointer(target_op)?)?;
        let linkpath = this.read_path_from_c_str(this.read_pointer(linkpath_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`symlink`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let result = create_link(&target, &linkpath).map(|_| 0);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn macos_fbsd_solarish_stat(
        &mut self,
        path_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !matches!(&*this.tcx.sess.target.os, "macos" | "freebsd" | "solaris" | "illumos") {
            panic!("`macos_fbsd_solaris_stat` should not be called on {}", this.tcx.sess.target.os);
        }

        let path_scalar = this.read_pointer(path_op)?;
        let path = this.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`stat`", reject_with)?;
            return this.set_last_error_and_return_i32(LibcError("EACCES"));
        }

        // `stat` always follows symlinks.
        let metadata = match FileMetadata::from_path(this, &path, true)? {
            Ok(metadata) => metadata,
            Err(err) => return this.set_last_error_and_return_i32(err),
        };

        interp_ok(Scalar::from_i32(this.macos_fbsd_solarish_write_stat_buf(metadata, buf_op)?))
    }

    // `lstat` is used to get symlink metadata.
    fn macos_fbsd_solarish_lstat(
        &mut self,
        path_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !matches!(&*this.tcx.sess.target.os, "macos" | "freebsd" | "solaris" | "illumos") {
            panic!(
                "`macos_fbsd_solaris_lstat` should not be called on {}",
                this.tcx.sess.target.os
            );
        }

        let path_scalar = this.read_pointer(path_op)?;
        let path = this.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`lstat`", reject_with)?;
            return this.set_last_error_and_return_i32(LibcError("EACCES"));
        }

        let metadata = match FileMetadata::from_path(this, &path, false)? {
            Ok(metadata) => metadata,
            Err(err) => return this.set_last_error_and_return_i32(err),
        };

        interp_ok(Scalar::from_i32(this.macos_fbsd_solarish_write_stat_buf(metadata, buf_op)?))
    }

    fn macos_fbsd_solarish_fstat(
        &mut self,
        fd_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !matches!(&*this.tcx.sess.target.os, "macos" | "freebsd" | "solaris" | "illumos") {
            panic!(
                "`macos_fbsd_solaris_fstat` should not be called on {}",
                this.tcx.sess.target.os
            );
        }

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fstat`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        let metadata = match FileMetadata::from_fd_num(this, fd)? {
            Ok(metadata) => metadata,
            Err(err) => return this.set_last_error_and_return_i32(err),
        };
        interp_ok(Scalar::from_i32(this.macos_fbsd_solarish_write_stat_buf(metadata, buf_op)?))
    }

    fn linux_statx(
        &mut self,
        dirfd_op: &OpTy<'tcx>,    // Should be an `int`
        pathname_op: &OpTy<'tcx>, // Should be a `const char *`
        flags_op: &OpTy<'tcx>,    // Should be an `int`
        mask_op: &OpTy<'tcx>,     // Should be an `unsigned int`
        statxbuf_op: &OpTy<'tcx>, // Should be a `struct statx *`
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "statx");

        let dirfd = this.read_scalar(dirfd_op)?.to_i32()?;
        let pathname_ptr = this.read_pointer(pathname_op)?;
        let flags = this.read_scalar(flags_op)?.to_i32()?;
        let _mask = this.read_scalar(mask_op)?.to_u32()?;
        let statxbuf_ptr = this.read_pointer(statxbuf_op)?;

        // If the statxbuf or pathname pointers are null, the function fails with `EFAULT`.
        if this.ptr_is_null(statxbuf_ptr)? || this.ptr_is_null(pathname_ptr)? {
            return this.set_last_error_and_return_i32(LibcError("EFAULT"));
        }

        let statxbuf = this.deref_pointer_as(statxbuf_op, this.libc_ty_layout("statx"))?;

        let path = this.read_path_from_c_str(pathname_ptr)?.into_owned();
        // See <https://github.com/rust-lang/rust/pull/79196> for a discussion of argument sizes.
        let at_empty_path = this.eval_libc_i32("AT_EMPTY_PATH");
        let empty_path_flag = flags & at_empty_path == at_empty_path;
        // We only support:
        // * interpreting `path` as an absolute directory,
        // * interpreting `path` as a path relative to `dirfd` when the latter is `AT_FDCWD`, or
        // * interpreting `dirfd` as any file descriptor when `path` is empty and AT_EMPTY_PATH is
        // set.
        // Other behaviors cannot be tested from `libstd` and thus are not implemented. If you
        // found this error, please open an issue reporting it.
        if !(path.is_absolute()
            || dirfd == this.eval_libc_i32("AT_FDCWD")
            || (path.as_os_str().is_empty() && empty_path_flag))
        {
            throw_unsup_format!(
                "using statx is only supported with absolute paths, relative paths with the file \
                descriptor `AT_FDCWD`, and empty paths with the `AT_EMPTY_PATH` flag set and any \
                file descriptor"
            )
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`statx`", reject_with)?;
            let ecode = if path.is_absolute() || dirfd == this.eval_libc_i32("AT_FDCWD") {
                // since `path` is provided, either absolute or
                // relative to CWD, `EACCES` is the most relevant.
                LibcError("EACCES")
            } else {
                // `dirfd` is set to target file, and `path` is empty
                // (or we would have hit the `throw_unsup_format`
                // above). `EACCES` would violate the spec.
                assert!(empty_path_flag);
                LibcError("EBADF")
            };
            return this.set_last_error_and_return_i32(ecode);
        }

        // the `_mask_op` parameter specifies the file information that the caller requested.
        // However `statx` is allowed to return information that was not requested or to not
        // return information that was requested. This `mask` represents the information we can
        // actually provide for any target.
        let mut mask = this.eval_libc_u32("STATX_TYPE") | this.eval_libc_u32("STATX_SIZE");

        // If the `AT_SYMLINK_NOFOLLOW` flag is set, we query the file's metadata without following
        // symbolic links.
        let follow_symlink = flags & this.eval_libc_i32("AT_SYMLINK_NOFOLLOW") == 0;

        // If the path is empty, and the AT_EMPTY_PATH flag is set, we query the open file
        // represented by dirfd, whether it's a directory or otherwise.
        let metadata = if path.as_os_str().is_empty() && empty_path_flag {
            FileMetadata::from_fd_num(this, dirfd)?
        } else {
            FileMetadata::from_path(this, &path, follow_symlink)?
        };
        let metadata = match metadata {
            Ok(metadata) => metadata,
            Err(err) => return this.set_last_error_and_return_i32(err),
        };

        // The `mode` field specifies the type of the file and the permissions over the file for
        // the owner, its group and other users. Given that we can only provide the file type
        // without using platform specific methods, we only set the bits corresponding to the file
        // type. This should be an `__u16` but `libc` provides its values as `u32`.
        let mode: u16 = metadata
            .mode
            .to_u32()?
            .try_into()
            .unwrap_or_else(|_| bug!("libc contains bad value for constant"));

        // We need to set the corresponding bits of `mask` if the access, creation and modification
        // times were available. Otherwise we let them be zero.
        let (access_sec, access_nsec) = metadata
            .accessed
            .map(|tup| {
                mask |= this.eval_libc_u32("STATX_ATIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        let (created_sec, created_nsec) = metadata
            .created
            .map(|tup| {
                mask |= this.eval_libc_u32("STATX_BTIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        let (modified_sec, modified_nsec) = metadata
            .modified
            .map(|tup| {
                mask |= this.eval_libc_u32("STATX_MTIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        // Now we write everything to `statxbuf`. We write a zero for the unavailable fields.
        this.write_int_fields_named(
            &[
                ("stx_mask", mask.into()),
                ("stx_blksize", 0),
                ("stx_attributes", 0),
                ("stx_nlink", 0),
                ("stx_uid", 0),
                ("stx_gid", 0),
                ("stx_mode", mode.into()),
                ("stx_ino", 0),
                ("stx_size", metadata.size.into()),
                ("stx_blocks", 0),
                ("stx_attributes_mask", 0),
                ("stx_rdev_major", 0),
                ("stx_rdev_minor", 0),
                ("stx_dev_major", 0),
                ("stx_dev_minor", 0),
            ],
            &statxbuf,
        )?;
        #[rustfmt::skip]
        this.write_int_fields_named(
            &[
                ("tv_sec", access_sec.into()),
                ("tv_nsec", access_nsec.into()),
            ],
            &this.project_field_named(&statxbuf, "stx_atime")?,
        )?;
        #[rustfmt::skip]
        this.write_int_fields_named(
            &[
                ("tv_sec", created_sec.into()),
                ("tv_nsec", created_nsec.into()),
            ],
            &this.project_field_named(&statxbuf, "stx_btime")?,
        )?;
        #[rustfmt::skip]
        this.write_int_fields_named(
            &[
                ("tv_sec", 0.into()),
                ("tv_nsec", 0.into()),
            ],
            &this.project_field_named(&statxbuf, "stx_ctime")?,
        )?;
        #[rustfmt::skip]
        this.write_int_fields_named(
            &[
                ("tv_sec", modified_sec.into()),
                ("tv_nsec", modified_nsec.into()),
            ],
            &this.project_field_named(&statxbuf, "stx_mtime")?,
        )?;

        interp_ok(Scalar::from_i32(0))
    }

    fn rename(
        &mut self,
        oldpath_op: &OpTy<'tcx>,
        newpath_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let oldpath_ptr = this.read_pointer(oldpath_op)?;
        let newpath_ptr = this.read_pointer(newpath_op)?;

        if this.ptr_is_null(oldpath_ptr)? || this.ptr_is_null(newpath_ptr)? {
            return this.set_last_error_and_return_i32(LibcError("EFAULT"));
        }

        let oldpath = this.read_path_from_c_str(oldpath_ptr)?;
        let newpath = this.read_path_from_c_str(newpath_ptr)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`rename`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let result = rename(oldpath, newpath).map(|_| 0);

        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn mkdir(&mut self, path_op: &OpTy<'tcx>, mode_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        #[cfg_attr(not(unix), allow(unused_variables))]
        let mode = if matches!(&*this.tcx.sess.target.os, "macos" | "freebsd") {
            u32::from(this.read_scalar(mode_op)?.to_u16()?)
        } else {
            this.read_scalar(mode_op)?.to_u32()?
        };

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`mkdir`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = DirBuilder::new();

        // If the host supports it, forward on the mode of the directory
        // (i.e. permission bits and the sticky bit)
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(mode);
        }

        let result = builder.create(path).map(|_| 0i32);

        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn rmdir(&mut self, path_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`rmdir`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let result = remove_dir(path).map(|_| 0i32);

        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    fn opendir(&mut self, name_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let name = this.read_path_from_c_str(this.read_pointer(name_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`opendir`", reject_with)?;
            this.set_last_error(LibcError("EACCES"))?;
            return interp_ok(Scalar::null_ptr(this));
        }

        let result = read_dir(name);

        match result {
            Ok(dir_iter) => {
                let id = this.machine.dirs.insert_new(dir_iter);

                // The libc API for opendir says that this method returns a pointer to an opaque
                // structure, but we are returning an ID number. Thus, pass it as a scalar of
                // pointer width.
                interp_ok(Scalar::from_target_usize(id, this))
            }
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(Scalar::null_ptr(this))
            }
        }
    }

    fn linux_solarish_readdir64(
        &mut self,
        dirent_type: &str,
        dirp_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !matches!(&*this.tcx.sess.target.os, "linux" | "solaris" | "illumos") {
            panic!("`linux_solaris_readdir64` should not be called on {}", this.tcx.sess.target.os);
        }

        let dirp = this.read_target_usize(dirp_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readdir`", reject_with)?;
            this.set_last_error(LibcError("EBADF"))?;
            return interp_ok(Scalar::null_ptr(this));
        }

        let open_dir = this.machine.dirs.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir64 did not come from opendir")
        })?;

        let entry = match open_dir.read_dir.next() {
            Some(Ok(dir_entry)) => {
                // Write the directory entry into a newly allocated buffer.
                // The name is written with write_bytes, while the rest of the
                // dirent64 (or dirent) struct is written using write_int_fields.

                // For reference:
                // On Linux:
                // pub struct dirent64 {
                //     pub d_ino: ino64_t,
                //     pub d_off: off64_t,
                //     pub d_reclen: c_ushort,
                //     pub d_type: c_uchar,
                //     pub d_name: [c_char; 256],
                // }
                //
                // On Solaris:
                // pub struct dirent {
                //     pub d_ino: ino64_t,
                //     pub d_off: off64_t,
                //     pub d_reclen: c_ushort,
                //     pub d_name: [c_char; 3],
                // }

                let mut name = dir_entry.file_name(); // not a Path as there are no separators!
                name.push("\0"); // Add a NUL terminator
                let name_bytes = name.as_encoded_bytes();
                let name_len = u64::try_from(name_bytes.len()).unwrap();

                let dirent_layout = this.libc_ty_layout(dirent_type);
                let fields = &dirent_layout.fields;
                let last_field = fields.count().strict_sub(1);
                let d_name_offset = fields.offset(last_field).bytes();
                let size = d_name_offset.strict_add(name_len);

                let entry = this.allocate_ptr(
                    Size::from_bytes(size),
                    dirent_layout.align.abi,
                    MiriMemoryKind::Runtime.into(),
                    AllocInit::Uninit,
                )?;
                let entry: Pointer = entry.into();

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;
                this.write_int_fields_named(
                    &[("d_ino", ino.into()), ("d_off", 0), ("d_reclen", size.into())],
                    &this.ptr_to_mplace(entry, dirent_layout),
                )?;

                if let Some(d_type) = this
                    .try_project_field_named(&this.ptr_to_mplace(entry, dirent_layout), "d_type")?
                {
                    this.write_int(file_type, &d_type)?;
                }

                let name_ptr = entry.wrapping_offset(Size::from_bytes(d_name_offset), this);
                this.write_bytes_ptr(name_ptr, name_bytes.iter().copied())?;

                Some(entry)
            }
            None => {
                // end of stream: return NULL
                None
            }
            Some(Err(e)) => {
                this.set_last_error(e)?;
                None
            }
        };

        let open_dir = this.machine.dirs.streams.get_mut(&dirp).unwrap();
        let old_entry = std::mem::replace(&mut open_dir.entry, entry);
        if let Some(old_entry) = old_entry {
            this.deallocate_ptr(old_entry, None, MiriMemoryKind::Runtime.into())?;
        }

        interp_ok(Scalar::from_maybe_pointer(entry.unwrap_or_else(Pointer::null), this))
    }

    fn macos_fbsd_readdir_r(
        &mut self,
        dirp_op: &OpTy<'tcx>,
        entry_op: &OpTy<'tcx>,
        result_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !matches!(&*this.tcx.sess.target.os, "macos" | "freebsd") {
            panic!("`macos_fbsd_readdir_r` should not be called on {}", this.tcx.sess.target.os);
        }

        let dirp = this.read_target_usize(dirp_op)?;
        let result_place = this.deref_pointer_as(result_op, this.machine.layouts.mut_raw_ptr)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readdir_r`", reject_with)?;
            // Return error code, do *not* set `errno`.
            return interp_ok(this.eval_libc("EBADF"));
        }

        let open_dir = this.machine.dirs.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir_r did not come from opendir")
        })?;
        interp_ok(match open_dir.read_dir.next() {
            Some(Ok(dir_entry)) => {
                // Write into entry, write pointer to result, return 0 on success.
                // The name is written with write_os_str_to_c_str, while the rest of the
                // dirent struct is written using write_int_fields.

                // For reference, on macOS this looks like:
                // pub struct dirent {
                //     pub d_ino: u64,
                //     pub d_seekoff: u64,
                //     pub d_reclen: u16,
                //     pub d_namlen: u16,
                //     pub d_type: u8,
                //     pub d_name: [c_char; 1024],
                // }

                let entry_place = this.deref_pointer_as(entry_op, this.libc_ty_layout("dirent"))?;
                let name_place = this.project_field_named(&entry_place, "d_name")?;

                let file_name = dir_entry.file_name(); // not a Path as there are no separators!
                let (name_fits, file_name_buf_len) = this.write_os_str_to_c_str(
                    &file_name,
                    name_place.ptr(),
                    name_place.layout.size.bytes(),
                )?;
                let file_name_len = file_name_buf_len.strict_sub(1);
                if !name_fits {
                    throw_unsup_format!(
                        "a directory entry had a name too large to fit in libc::dirent"
                    );
                }

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;

                // Common fields.
                this.write_int_fields_named(
                    &[
                        ("d_reclen", 0),
                        ("d_namlen", file_name_len.into()),
                        ("d_type", file_type.into()),
                    ],
                    &entry_place,
                )?;
                // Special fields.
                match &*this.tcx.sess.target.os {
                    "macos" => {
                        #[rustfmt::skip]
                        this.write_int_fields_named(
                            &[
                                ("d_ino", ino.into()),
                                ("d_seekoff", 0),
                            ],
                            &entry_place,
                        )?;
                    }
                    "freebsd" => {
                        #[rustfmt::skip]
                        this.write_int_fields_named(
                            &[
                                ("d_fileno", ino.into()),
                                ("d_off", 0),
                            ],
                            &entry_place,
                        )?;
                    }
                    _ => unreachable!(),
                }
                this.write_scalar(this.read_scalar(entry_op)?, &result_place)?;

                Scalar::from_i32(0)
            }
            None => {
                // end of stream: return 0, assign *result=NULL
                this.write_null(&result_place)?;
                Scalar::from_i32(0)
            }
            Some(Err(e)) => {
                // return positive error number on error (do *not* set last error)
                this.io_error_to_errnum(e)?
            }
        })
    }

    fn closedir(&mut self, dirp_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let dirp = this.read_target_usize(dirp_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`closedir`", reject_with)?;
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        let Some(mut open_dir) = this.machine.dirs.streams.remove(&dirp) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        if let Some(entry) = open_dir.entry.take() {
            this.deallocate_ptr(entry, None, MiriMemoryKind::Runtime.into())?;
        }
        // We drop the `open_dir`, which will close the host dir handle.
        drop(open_dir);

        interp_ok(Scalar::from_i32(0))
    }

    fn ftruncate64(&mut self, fd_num: i32, length: i128) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`ftruncate64`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        let Some(fd) = this.machine.fds.get(fd_num) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };

        // FIXME: Support ftruncate64 for all FDs
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`ftruncate64` is only supported on file-backed file descriptors")
        })?;

        if file.writable {
            if let Ok(length) = length.try_into() {
                let result = file.file.set_len(length);
                let result = this.try_unwrap_io_result(result.map(|_| 0i32))?;
                interp_ok(Scalar::from_i32(result))
            } else {
                this.set_last_error_and_return_i32(LibcError("EINVAL"))
            }
        } else {
            // The file is not writable
            this.set_last_error_and_return_i32(LibcError("EINVAL"))
        }
    }

    fn fsync(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        // On macOS, `fsync` (unlike `fcntl(F_FULLFSYNC)`) does not wait for the
        // underlying disk to finish writing. In the interest of host compatibility,
        // we conservatively implement this with `sync_all`, which
        // *does* wait for the disk.

        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fsync`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        self.ffullsync_fd(fd)
    }

    fn ffullsync_fd(&mut self, fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let Some(fd) = this.machine.fds.get(fd_num) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`fsync` is only supported on file-backed file descriptors")
        })?;
        let io_result = maybe_sync_file(&file.file, file.writable, File::sync_all);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(io_result)?))
    }

    fn fdatasync(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fdatasync`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        let Some(fd) = this.machine.fds.get(fd) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`fdatasync` is only supported on file-backed file descriptors")
        })?;
        let io_result = maybe_sync_file(&file.file, file.writable, File::sync_data);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(io_result)?))
    }

    fn sync_file_range(
        &mut self,
        fd_op: &OpTy<'tcx>,
        offset_op: &OpTy<'tcx>,
        nbytes_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let offset = this.read_scalar(offset_op)?.to_i64()?;
        let nbytes = this.read_scalar(nbytes_op)?.to_i64()?;
        let flags = this.read_scalar(flags_op)?.to_i32()?;

        if offset < 0 || nbytes < 0 {
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
        }
        let allowed_flags = this.eval_libc_i32("SYNC_FILE_RANGE_WAIT_BEFORE")
            | this.eval_libc_i32("SYNC_FILE_RANGE_WRITE")
            | this.eval_libc_i32("SYNC_FILE_RANGE_WAIT_AFTER");
        if flags & allowed_flags != flags {
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`sync_file_range`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        }

        let Some(fd) = this.machine.fds.get(fd) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`sync_data_range` is only supported on file-backed file descriptors")
        })?;
        let io_result = maybe_sync_file(&file.file, file.writable, File::sync_data);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(io_result)?))
    }

    fn readlink(
        &mut self,
        pathname_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
        bufsize_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        let pathname = this.read_path_from_c_str(this.read_pointer(pathname_op)?)?;
        let buf = this.read_pointer(buf_op)?;
        let bufsize = this.read_target_usize(bufsize_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readlink`", reject_with)?;
            this.set_last_error(LibcError("EACCES"))?;
            return interp_ok(-1);
        }

        let result = std::fs::read_link(pathname);
        match result {
            Ok(resolved) => {
                // 'readlink' truncates the resolved path if the provided buffer is not large
                // enough, and does *not* add a null terminator. That means we cannot use the usual
                // `write_path_to_c_str` and have to re-implement parts of it ourselves.
                let resolved = this.convert_path(
                    Cow::Borrowed(resolved.as_ref()),
                    crate::shims::os_str::PathConversion::HostToTarget,
                );
                let mut path_bytes = resolved.as_encoded_bytes();
                let bufsize: usize = bufsize.try_into().unwrap();
                if path_bytes.len() > bufsize {
                    path_bytes = &path_bytes[..bufsize]
                }
                this.write_bytes_ptr(buf, path_bytes.iter().copied())?;
                interp_ok(path_bytes.len().try_into().unwrap())
            }
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(-1)
            }
        }
    }

    fn isatty(&mut self, miri_fd: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        // "returns 1 if fd is an open file descriptor referring to a terminal;
        // otherwise 0 is returned, and errno is set to indicate the error"
        let fd = this.read_scalar(miri_fd)?.to_i32()?;
        let error = if let Some(fd) = this.machine.fds.get(fd) {
            if fd.is_tty(this.machine.communicate()) {
                return interp_ok(Scalar::from_i32(1));
            } else {
                LibcError("ENOTTY")
            }
        } else {
            // FD does not exist
            LibcError("EBADF")
        };
        this.set_last_error(error)?;
        interp_ok(Scalar::from_i32(0))
    }

    fn realpath(
        &mut self,
        path_op: &OpTy<'tcx>,
        processed_path_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("realpath");

        let pathname = this.read_path_from_c_str(this.read_pointer(path_op)?)?;
        let processed_ptr = this.read_pointer(processed_path_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`realpath`", reject_with)?;
            this.set_last_error(LibcError("EACCES"))?;
            return interp_ok(Scalar::from_target_usize(0, this));
        }

        let result = std::fs::canonicalize(pathname);
        match result {
            Ok(resolved) => {
                let path_max = this
                    .eval_libc_i32("PATH_MAX")
                    .try_into()
                    .expect("PATH_MAX does not fit in u64");
                let dest = if this.ptr_is_null(processed_ptr)? {
                    // POSIX says behavior when passing a null pointer is implementation-defined,
                    // but GNU/linux, freebsd, netbsd, bionic/android, and macos all treat a null pointer
                    // similarly to:
                    //
                    // "If resolved_path is specified as NULL, then realpath() uses
                    // malloc(3) to allocate a buffer of up to PATH_MAX bytes to hold
                    // the resolved pathname, and returns a pointer to this buffer.  The
                    // caller should deallocate this buffer using free(3)."
                    // <https://man7.org/linux/man-pages/man3/realpath.3.html>
                    this.alloc_path_as_c_str(&resolved, MiriMemoryKind::C.into())?
                } else {
                    let (wrote_path, _) =
                        this.write_path_to_c_str(&resolved, processed_ptr, path_max)?;

                    if !wrote_path {
                        // Note that we do not explicitly handle `FILENAME_MAX`
                        // (different from `PATH_MAX` above) as it is Linux-specific and
                        // seems like a bit of a mess anyway: <https://eklitzke.org/path-max-is-tricky>.
                        this.set_last_error(LibcError("ENAMETOOLONG"))?;
                        return interp_ok(Scalar::from_target_usize(0, this));
                    }
                    processed_ptr
                };

                interp_ok(Scalar::from_maybe_pointer(dest, this))
            }
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(Scalar::from_target_usize(0, this))
            }
        }
    }
    fn mkstemp(&mut self, template_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        use rand::seq::IndexedRandom;

        // POSIX defines the template string.
        const TEMPFILE_TEMPLATE_STR: &str = "XXXXXX";

        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("mkstemp");

        // POSIX defines the maximum number of attempts before failure.
        //
        // `mkstemp()` relies on `tmpnam()` which in turn relies on `TMP_MAX`.
        // POSIX says this about `TMP_MAX`:
        // * Minimum number of unique filenames generated by `tmpnam()`.
        // * Maximum number of times an application can call `tmpnam()` reliably.
        //   * The value of `TMP_MAX` is at least 25.
        //   * On XSI-conformant systems, the value of `TMP_MAX` is at least 10000.
        // See <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/stdio.h.html>.
        let max_attempts = this.eval_libc_u32("TMP_MAX");

        // Get the raw bytes from the template -- as a byte slice, this is a string in the target
        // (and the target is unix, so a byte slice is the right representation).
        let template_ptr = this.read_pointer(template_op)?;
        let mut template = this.eval_context_ref().read_c_str(template_ptr)?.to_owned();
        let template_bytes = template.as_mut_slice();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`mkstemp`", reject_with)?;
            return this.set_last_error_and_return_i32(LibcError("EACCES"));
        }

        // Get the bytes of the suffix we expect in _target_ encoding.
        let suffix_bytes = TEMPFILE_TEMPLATE_STR.as_bytes();

        // At this point we have one `&[u8]` that represents the template and one `&[u8]`
        // that represents the expected suffix.

        // Now we figure out the index of the slice we expect to contain the suffix.
        let start_pos = template_bytes.len().saturating_sub(suffix_bytes.len());
        let end_pos = template_bytes.len();
        let last_six_char_bytes = &template_bytes[start_pos..end_pos];

        // If we don't find the suffix, it is an error.
        if last_six_char_bytes != suffix_bytes {
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
        }

        // At this point we know we have 6 ASCII 'X' characters as a suffix.

        // From <https://github.com/lattera/glibc/blob/895ef79e04a953cac1493863bcae29ad85657ee1/sysdeps/posix/tempname.c#L175>
        const SUBSTITUTIONS: &[char; 62] = &[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
            'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        ];

        // The file is opened with specific options, which Rust does not expose in a portable way.
        // So we use specific APIs depending on the host OS.
        let mut fopts = OpenOptions::new();
        fopts.read(true).write(true).create_new(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Do not allow others to read or modify this file.
            fopts.mode(0o600);
            fopts.custom_flags(libc::O_EXCL);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Do not allow others to read or modify this file.
            fopts.share_mode(0);
        }

        // If the generated file already exists, we will try again `max_attempts` many times.
        for _ in 0..max_attempts {
            let rng = this.machine.rng.get_mut();

            // Generate a random unique suffix.
            let unique_suffix = SUBSTITUTIONS.choose_multiple(rng, 6).collect::<String>();

            // Replace the template string with the random string.
            template_bytes[start_pos..end_pos].copy_from_slice(unique_suffix.as_bytes());

            // Write the modified template back to the passed in pointer to maintain POSIX semantics.
            this.write_bytes_ptr(template_ptr, template_bytes.iter().copied())?;

            // To actually open the file, turn this into a host OsString.
            let p = bytes_to_os_str(template_bytes)?.to_os_string();

            let possibly_unique = std::env::temp_dir().join::<PathBuf>(p.into());

            let file = fopts.open(possibly_unique);

            match file {
                Ok(f) => {
                    let fd = this.machine.fds.insert_new(FileHandle { file: f, writable: true });
                    return interp_ok(Scalar::from_i32(fd));
                }
                Err(e) =>
                    match e.kind() {
                        // If the random file already exists, keep trying.
                        ErrorKind::AlreadyExists => continue,
                        // Any other errors are returned to the caller.
                        _ => {
                            // "On error, -1 is returned, and errno is set to
                            // indicate the error"
                            return this.set_last_error_and_return_i32(e);
                        }
                    },
            }
        }

        // We ran out of attempts to create the file, return an error.
        this.set_last_error_and_return_i32(LibcError("EEXIST"))
    }
}

/// Extracts the number of seconds and nanoseconds elapsed between `time` and the unix epoch when
/// `time` is Ok. Returns `None` if `time` is an error. Fails if `time` happens before the unix
/// epoch.
fn extract_sec_and_nsec<'tcx>(
    time: std::io::Result<SystemTime>,
) -> InterpResult<'tcx, Option<(u64, u32)>> {
    match time.ok() {
        Some(time) => {
            let duration = system_time_to_duration(&time)?;
            interp_ok(Some((duration.as_secs(), duration.subsec_nanos())))
        }
        None => interp_ok(None),
    }
}

/// Stores a file's metadata in order to avoid code duplication in the different metadata related
/// shims.
struct FileMetadata {
    mode: Scalar,
    size: u64,
    created: Option<(u64, u32)>,
    accessed: Option<(u64, u32)>,
    modified: Option<(u64, u32)>,
    dev: u64,
    uid: u32,
    gid: u32,
}

impl FileMetadata {
    fn from_path<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        path: &Path,
        follow_symlink: bool,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        let metadata =
            if follow_symlink { std::fs::metadata(path) } else { std::fs::symlink_metadata(path) };

        FileMetadata::from_meta(ecx, metadata)
    }

    fn from_fd_num<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        fd_num: i32,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return interp_ok(Err(LibcError("EBADF")));
        };

        let metadata = fd.metadata()?;
        drop(fd);
        FileMetadata::from_meta(ecx, metadata)
    }

    fn from_meta<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        metadata: Result<std::fs::Metadata, std::io::Error>,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        let metadata = match metadata {
            Ok(metadata) => metadata,
            Err(e) => {
                return interp_ok(Err(e.into()));
            }
        };

        let file_type = metadata.file_type();

        let mode_name = if file_type.is_file() {
            "S_IFREG"
        } else if file_type.is_dir() {
            "S_IFDIR"
        } else {
            "S_IFLNK"
        };

        let mode = ecx.eval_libc(mode_name);

        let size = metadata.len();

        let created = extract_sec_and_nsec(metadata.created())?;
        let accessed = extract_sec_and_nsec(metadata.accessed())?;
        let modified = extract_sec_and_nsec(metadata.modified())?;

        // FIXME: Provide more fields using platform specific methods.

        cfg_select! {
            unix => {
                use std::os::unix::fs::MetadataExt;
                let dev = metadata.dev();
                let uid = metadata.uid();
                let gid = metadata.gid();
            }
            _ => {
                let dev = 0;
                let uid = 0;
                let gid = 0;
            }
        }

        interp_ok(Ok(FileMetadata { mode, size, created, accessed, modified, dev, uid, gid }))
    }
}
