//! File and file system access

use std::borrow::Cow;
use std::ffi::OsString;
use std::fs::{self, DirBuilder, File, FileTimes, OpenOptions, TryLockError};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{self, Path};
use std::time::SystemTime;

use rand::RngExt;
use rustc_abi::Size;
use rustc_data_structures::either::Either;
use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::Os;

use self::shims::time::system_time_to_duration;
use crate::helpers::OptionExt;
use crate::shims::files::{
    BIG_ACCESS, DirHandle, DynFileDescriptionRef, EvalContextExt as _, FdNum, FileHandle,
    open_file_or_dir,
};
use crate::shims::os_str::{PathConversion, bytes_to_os_str};
use crate::shims::sig::Varargs;
use crate::shims::unix::fd::{EvalContextExt as _, FlockOp, UnixFileDescription};
use crate::shims::{FdId, FileDescription, FileDescriptionRef};
use crate::*;

/// An open directory stream (`DIR`), tracked by DirTable.
#[derive(Debug)]
struct DirStream {
    /// The directory stream on the host.
    read_dir: fs::ReadDir,
    /// Unix directory streams have an "underlying FD" that can be exposed; this is that FD. This is
    /// not a reference; it *can* be separately closed, leading to UB if the DIR is used again. We
    /// also store the FD ID to catch cases where the FD was closed and a different one re-opened.
    fd_num: FdNum,
    fd_id: FdId,
    /// The "special" entries that must still be yielded by the iterator.
    /// Used for `.` and `..`.
    special_entries: Vec<&'static str>,
    /// The most recent entry returned by readdir(). Will be freed by the next call.
    /// The string is the type used to allocate it.
    entry: Option<(Pointer, &'static str)>,
}

impl DirStream {
    fn new(read_dir: fs::ReadDir, fd_num: FdNum, fd_id: FdId) -> Self {
        Self { read_dir, fd_num, fd_id, special_entries: vec!["..", "."], entry: None }
    }

    fn next_host_entry(&mut self) -> Option<io::Result<Either<fs::DirEntry, &'static str>>> {
        if let Some(special) = self.special_entries.pop() {
            return Some(Ok(Either::Right(special)));
        }
        let entry = self.read_dir.next()?;
        Some(entry.map(Either::Left))
    }
}

#[derive(Debug)]
struct DirEntry {
    name: OsString,
    ino: u64,
    d_type: i32,
}

/// What a `futimens` `timespec` asks for: leave the timestamp alone (`UTIME_OMIT`) or set it.
#[derive(Copy, Clone)]
enum TimeUpdate {
    Omit,
    Set(SystemTime),
}

impl UnixFileDescription for FileHandle {
    fn pread<'tcx>(
        &self,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        offset: u64,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        if !self.readable {
            return finish.call(ecx, Err(LibcError("EBADF")));
        }

        let file = &mut &self.file;
        // Emulates pread using seek + read + seek to restore cursor position.
        // Correctness of this emulation relies on sequential nature of Miri execution.
        let result = ecx.read_from_host(ptr, len, |buf| {
            let cursor_pos = file.stream_position()?;
            file.seek(SeekFrom::Start(offset))?;
            let res = file.read(buf);
            // Attempt to restore cursor position even if the read has failed
            file.seek(SeekFrom::Start(cursor_pos))
                .expect("failed to restore file position, this shouldn't be possible");
            res
        })?;
        finish.call(ecx, result)
    }

    fn pwrite<'tcx>(
        &self,
        communicate_allowed: bool,
        ptr: Pointer,
        len: u64,
        offset: u64,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<u64, IoError>>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        if !self.writable {
            return finish.call(ecx, Err(LibcError("EBADF")));
        }
        if self.append {
            // On "append" files, pwrite cannot be emulated with seeking.
            throw_unsup_format!("pwrite is not supported on files opened with O_APPEND");
        }

        let file = &mut &self.file;
        // Emulates pwrite using seek + write + seek to restore cursor position.
        // Correctness of this emulation relies on sequential nature of Miri execution.
        let result = ecx.write_to_host(ptr, len, |buf| {
            let cursor_pos = file.stream_position()?;
            file.seek(SeekFrom::Start(offset))?;
            let res = file.write(buf);
            // Attempt to restore cursor position even if the write has failed
            file.seek(SeekFrom::Start(cursor_pos))
                .expect("failed to restore file position, this shouldn't be possible");
            res
        })?;
        finish.call(ecx, result)
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

    fn get_flags<'tcx>(&self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        interp_ok(match (self.readable, self.writable) {
            (true, true) => ecx.eval_libc("O_RDWR"),
            (true, false) => ecx.eval_libc("O_RDONLY"),
            (false, true) => ecx.eval_libc("O_WRONLY"),
            _ => unreachable!(),
        })
    }
}

impl UnixFileDescription for DirHandle {
    fn get_flags<'tcx>(&self, ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, Scalar> {
        // Directories are always readonly on Unix.
        interp_ok(ecx.eval_libc("O_RDONLY"))
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
    streams: FxHashMap<u64, DirStream>,
    /// ID number to be used by the next call to opendir
    next_id: u64,
}

impl DirTable {
    #[expect(clippy::arithmetic_side_effects)]
    fn insert_new(&mut self, stream: DirStream) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.streams.try_insert(id, stream).unwrap();
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
    file: &FileHandle,
    operation: fn(&File) -> std::io::Result<()>,
) -> std::io::Result<i32> {
    if !file.writable && cfg!(windows) {
        // sync_all() and sync_data() will return an error on Windows hosts if the file is not opened
        // for writing. (FlushFileBuffers requires that the file handle have the
        // GENERIC_WRITE right)
        Ok(0i32)
    } else {
        let result = operation(&file.file);
        result.map(|_| 0i32)
    }
}

/// Whether `path` is a symlink to a directory on the host.
/// If `dir` is given, a relative `path` is resolved relative to that directory.
#[cfg(windows)]
fn host_is_symlink_dir(dir: Option<&DirHandle>, path: &Path) -> bool {
    use std::os::windows::fs::FileTypeExt;
    let metadata = match dir {
        Some(dir) => dir.dir.symlink_metadata(path),
        None => path.symlink_metadata(),
    };
    metadata.is_ok_and(|m| m.file_type().is_symlink_dir())
}

/// Removes the file at `path` on the host, with the semantics of `unlink`.
/// If `dir` is given, a relative `path` is resolved relative to that directory.
fn host_remove_file(dir: Option<&DirHandle>, path: &Path) -> io::Result<()> {
    cfg_select! {
        unix =>
            match dir {
                Some(dir) => dir.dir.remove_file(path),
                None => fs::remove_file(path),
            },
        windows => {{
            // This should be able to remove symlinks to dirs, but on Windows that requires
            // `remove_dir`. Our work-around is racy but there's not a lot we can do about that.
            // FIXME: maybe we can retry based on the error code?
            let symlink_dir = host_is_symlink_dir(dir, path);
            match (dir, symlink_dir) {
                (Some(dir), true) => dir.dir.remove_dir(path),
                (Some(dir), false) => dir.dir.remove_file(path),
                (None, true) => fs::remove_dir(path),
                (None, false) => fs::remove_file(path),
            }
        }}
    }
}

/// Removes the directory at `path` on the host, with the semantics of `rmdir`.
/// If `dir` is given, a relative `path` is resolved relative to that directory.
fn host_remove_dir(dir: Option<&DirHandle>, path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        // This should *not* be able to remove symlinks to dirs, but on Windows it is. Our
        // work-around is racy (it might change to being a directory symlink after we
        // checked it), but there's not a lot we can do about that.
        if host_is_symlink_dir(dir, path) {
            return Err(ErrorKind::NotADirectory.into());
        }
    }
    match dir {
        Some(dir) => dir.dir.remove_dir(path),
        None => fs::remove_dir(path),
    }
}

/// The host `FileTimes` that applies the given access and modification time updates.
fn host_file_times_from_updates(access: TimeUpdate, modified: TimeUpdate) -> FileTimes {
    let mut filetimes = FileTimes::new();
    if let TimeUpdate::Set(access) = access {
        filetimes = filetimes.set_accessed(access);
    }
    if let TimeUpdate::Set(modified) = modified {
        filetimes = filetimes.set_modified(modified);
    }
    filetimes
}

fn same_host_file(file1: &fs::Metadata, file2: &fs::Metadata) -> bool {
    cfg_select! {
        unix => {
            use std::os::unix::fs::MetadataExt;
            file1.dev() == file2.dev() && file1.ino() == file2.ino()
        }
        windows => {
            use std::os::windows::fs::MetadataExt;
            file1.volume_serial_number().unwrap() == file2.volume_serial_number().unwrap()
                && file1.file_index().unwrap() == file2.file_index().unwrap()
        }
    }
}

impl<'tcx> EvalContextExtPrivate<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPrivate<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Decode one `futimens` `timespec`, handling the `UTIME_NOW`/`UTIME_OMIT` `tv_nsec` values.
    /// `None` means the `timespec` is invalid and the caller should report `EINVAL`.
    fn parse_utimens_timespec(
        &self,
        tp: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, Option<TimeUpdate>> {
        let ecx = self.eval_context_ref();
        // `UTIME_NOW` reads the host clock, which we must not do under isolation.
        assert!(ecx.machine.communicate(), "isolation should have prevented reaching this");

        // `tv_nsec` and the `UTIME_*` constants are `c_long`, i.e. the target's `isize`.
        let nsec_place = ecx.project_field_named(tp, "tv_nsec")?;
        let nsec = ecx.read_scalar(&nsec_place)?.to_target_isize(ecx)?;

        if nsec == ecx.eval_libc("UTIME_OMIT").to_target_isize(ecx)? {
            return interp_ok(Some(TimeUpdate::Omit));
        }
        if nsec == ecx.eval_libc("UTIME_NOW").to_target_isize(ecx)? {
            return interp_ok(Some(TimeUpdate::Set(SystemTime::now())));
        }

        let Some(duration) = ecx.read_timespec(tp)? else {
            return interp_ok(None);
        };
        interp_ok(SystemTime::UNIX_EPOCH.checked_add(duration).map(TimeUpdate::Set))
    }

    /// Decode the `times` argument of `futimens`/`utimensat`: `[atime, mtime]`, or NULL to set both
    /// to now, as `(access, modified)`.
    /// `None` means a `timespec` is invalid and the caller should report `EINVAL`.
    fn parse_utimens_times(
        &self,
        times: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, Option<(TimeUpdate, TimeUpdate)>> {
        let ecx = self.eval_context_ref();

        if ecx.ptr_is_null(times.ptr())? {
            let now = TimeUpdate::Set(SystemTime::now());
            interp_ok(Some((now, now)))
        } else {
            let access_place = ecx.project_c_array_field(times, 0)?;
            let modified_place = ecx.project_c_array_field(times, 1)?;
            let Some(access) = ecx.parse_utimens_timespec(&access_place)? else {
                return interp_ok(None);
            };
            let Some(modified) = ecx.parse_utimens_timespec(&modified_place)? else {
                return interp_ok(None);
            };
            interp_ok(Some((access, modified)))
        }
    }

    fn write_stat_buf(
        &mut self,
        metadata: FileMetadata,
        buf_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        let ecx = self.eval_context_mut();

        let (access_sec, access_nsec) = metadata.accessed.unwrap_or((0, 0));
        let (created_sec, created_nsec) = metadata.created.unwrap_or((0, 0));
        let (modified_sec, modified_nsec) = metadata.modified.unwrap_or((0, 0));

        let buf = ecx.deref_pointer(buf_op)?;

        ecx.write_int_fields_named(
            &[
                ("st_dev", metadata.dev.unwrap_or(0).into()),
                ("st_mode", metadata.mode.into()),
                ("st_nlink", metadata.nlink.unwrap_or(0).into()),
                ("st_ino", metadata.ino.unwrap_or(0).into()),
                ("st_uid", metadata.uid.unwrap_or(0).into()),
                ("st_gid", metadata.gid.unwrap_or(0).into()),
                ("st_rdev", 0),
                ("st_atime", access_sec.into()),
                ("st_atime_nsec", access_nsec.into()),
                ("st_mtime", modified_sec.into()),
                ("st_mtime_nsec", modified_nsec.into()),
                ("st_ctime", 0),
                ("st_ctime_nsec", 0),
                ("st_size", metadata.size.into()),
                ("st_blocks", metadata.blocks.unwrap_or(0).into()),
                ("st_blksize", metadata.blksize.unwrap_or(0).into()),
            ],
            &buf,
        )?;

        if matches!(&ecx.tcx.sess.target.os, Os::MacOs | Os::FreeBsd) {
            ecx.write_int_fields_named(
                &[
                    ("st_birthtime", created_sec.into()),
                    ("st_birthtime_nsec", created_nsec.into()),
                    ("st_flags", 0),
                    ("st_gen", 0),
                ],
                &buf,
            )?;
        }

        if matches!(&ecx.tcx.sess.target.os, Os::Solaris | Os::Illumos) {
            let st_fstype = ecx.project_field_named(&buf, "st_fstype")?;
            // This is an array; write 0 into first element so that it encodes the empty string.
            ecx.write_int(0, &ecx.project_index(&st_fstype, 0)?)?;
        }

        interp_ok(0)
    }

    #[cfg(windows)]
    fn file_type_to_d_type(&self, file_type: fs::FileType) -> InterpResult<'tcx, i32> {
        let ecx = self.eval_context_ref();
        match () {
            _ if file_type.is_dir() => interp_ok(ecx.eval_libc("DT_DIR").to_u8()?.into()),
            _ if file_type.is_file() => interp_ok(ecx.eval_libc("DT_REG").to_u8()?.into()),
            _ if file_type.is_symlink() => interp_ok(ecx.eval_libc("DT_LNK").to_u8()?.into()),
            // Fallback
            _ => interp_ok(ecx.eval_libc("DT_UNKNOWN").to_u8()?.into()),
        }
    }

    fn dir_entry_fields(
        &self,
        entry: Either<fs::DirEntry, &'static str>,
    ) -> InterpResult<'tcx, DirEntry> {
        let ecx = self.eval_context_ref();
        interp_ok(match entry {
            Either::Left(dir_entry) => {
                cfg_select! {
                    unix => {
                        use std::os::unix::fs::DirEntryExt;
                        DirEntry {
                            name: dir_entry.file_name(),
                            d_type: {
                                // We cannot rely on `dir_entry.file_type`, it might use a separate
                                // syscall based on `file_name` which is racy.
                                // It's always okay to return DT_UNKNOWN instead.
                                ecx.eval_libc("DT_UNKNOWN").to_u8()?.into()
                            },
                            ino: dir_entry.ino(),
                        }
                    }
                    windows => {
                        DirEntry {
                            name: dir_entry.file_name(),
                            // `DirEntry::file_type` is always race-free and never fails on Windows.
                            d_type: ecx.file_type_to_d_type(dir_entry.file_type().unwrap())?,
                            ino: 0u64,
                        }
                    }
                }
            }
            Either::Right(special) =>
                DirEntry {
                    name: special.into(),
                    d_type: ecx.eval_libc("DT_DIR").to_u8()?.into(),
                    ino: 0,
                },
        })
    }

    fn host_permissions_from_mode(
        &self,
        mode: u32,
        old_permissions: impl FnOnce() -> io::Result<fs::Permissions>,
    ) -> InterpResult<'tcx, io::Result<fs::Permissions>> {
        cfg_select! {
            unix => {
                use std::os::unix::fs::PermissionsExt;

                let _ = old_permissions;
                interp_ok(Ok(fs::Permissions::from_mode(mode)))
            }
            _ => {
                let ecx = self.eval_context_ref();

                let mut permissions = match old_permissions() {
                    Ok(p) => p,
                    Err(err) => return interp_ok(Err(err)),
                };
                // Only support default mode for non-UNIX (i.e. Windows) host.
                // Ignore the "regular file" bit which is a NOP.
                let s_ifreg = ecx.eval_libc_int::<u32>("S_IFREG");
                match mode & !s_ifreg {
                    0o666 => {
                        permissions.set_readonly(false);
                    }
                    0o444 => {
                        permissions.set_readonly(true);
                    }
                    _ => {
                        throw_unsup_format!(
                            "changing permissions to mode 0o{:o} is not supported on non-Unix hosts",
                            mode
                        );
                    }
                }
                interp_ok(Ok(permissions))
            }
        }
    }

    fn get_dir(
        &self,
        dirfd: Option<FdNum>,
        path: &Path,
    ) -> Result<Option<FileDescriptionRef<DirHandle>>, IoError> {
        let ecx = self.eval_context_ref();
        if path.is_empty() {
            Err(LibcError("ENOENT"))
        } else if path.is_absolute()
            || dirfd.is_none_or(|dirfd| dirfd == ecx.eval_libc_i32("AT_FDCWD"))
        {
            // It would be nice to make everything uniform and simply return `fs::Dir::open(".")`
            // here. But on Windows that would behave very different than methods on `fs`, e.g.
            // `..` would not work, so we cannot do that.
            Ok(None)
        } else {
            // relative to dirfd, which must be a directory handle
            let Some(fd) = ecx.machine.fds.get(dirfd.unwrap()) else {
                return Err(LibcError("EBADF"));
            };
            let Some(dir) = fd.downcast::<DirHandle>() else {
                return Err(LibcError("ENOTDIR"));
            };
            Ok(Some(dir))
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn open(
        &mut self,
        dirfd: Option<&OpTy<'tcx>>,
        path_raw: &OpTy<'tcx>,
        flag: &OpTy<'tcx>,
        varargs: Varargs<'tcx, '_>,
    ) -> InterpResult<'tcx, Scalar> {
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;
        #[cfg(windows)]
        use std::os::windows::fs::OpenOptionsExt;

        let ecx = self.eval_context_mut();
        let open_name = if dirfd.is_some() { "openat" } else { "open" };

        let dirfd = dirfd.try_map(|dirfd| ecx.read_scalar(dirfd)?.to_i32())?;
        let path_raw = ecx.read_pointer(path_raw)?;
        let flag = ecx.read_scalar(flag)?.to_i32()?;

        let path = ecx.read_path_from_c_str(path_raw)?;
        // Files in `/proc` won't work properly.
        if matches!(ecx.tcx.sess.target.os, Os::Linux | Os::Android | Os::Illumos | Os::Solaris)
            && path::absolute(&path).is_ok_and(|path| path.starts_with("/proc"))
        {
            ecx.machine.emit_diagnostic(NonHaltingDiagnostic::FileInProcOpened);
        }

        // Helper for those flags that read the `mode` parameter.
        let read_mode = |ecx: &MiriInterpCx<'tcx>,
                         options: &mut OpenOptions,
                         fn_name: &str|
         -> InterpResult<'tcx> {
            let ([mode], _) = ecx.check_varargs(
                if ecx.libc_ty_layout("mode_t").size.bytes() >= 4 {
                    // `mode_t` is big enough, no C integer promotion.
                    shim_varargs![libc::mode_t]
                } else {
                    // Types smaller than int get promoted to int
                    // (see https://github.com/rust-lang/rust/issues/71915).
                    shim_varargs![i32]
                },
                varargs,
                fn_name,
            )?;
            let mode = ecx.read_scalar(&mode)?.to_u32()?;

            cfg_select! {
                unix => {
                    // Support all modes on UNIX host
                    options.mode(mode);
                }
                _ => {
                    let _unused = options;

                    // Only support default mode for non-UNIX (i.e. Windows) host.
                    // Ignore the "regular file" bit which is a NOP.
                    let s_ifreg = ecx.eval_libc_int::<u32>("S_IFREG");
                    if mode & !s_ifreg != 0o666 {
                        throw_unsup_format!(
                            "{open_name}: non-default mode 0o{:o} is not supported on non-Unix hosts",
                            mode
                        );
                    }
                }
            }

            interp_ok(())
        };

        // We will "subtract" supported flags from this and at the end check that no bits are left.
        let mut flag = flag;

        let mut options = OpenOptions::new();
        let mut custom_flags = 0;

        let o_rdonly = ecx.eval_libc_i32("O_RDONLY");
        let o_wronly = ecx.eval_libc_i32("O_WRONLY");
        let o_rdwr = ecx.eval_libc_i32("O_RDWR");
        // The first two bits of the flag correspond to the access mode in linux, macOS and
        // windows. We need to check that in fact the access mode flags for the current target
        // only use these two bits, otherwise we are in an unsupported target and should error.
        if (o_rdonly | o_wronly | o_rdwr) & !0b11 != 0 {
            throw_unsup_format!("{open_name}: access mode flags on this target are unsupported");
        }
        let mut writable = true;
        let mut readable = true;

        // Now we check the access mode
        let access_mode = flag & ecx.eval_libc_i32("O_ACCMODE");
        flag &= !access_mode;

        if access_mode == o_rdonly {
            writable = false;
            options.read(true);
        } else if access_mode == o_wronly {
            readable = false;
            options.write(true);
        } else if access_mode == o_rdwr {
            options.read(true).write(true);
        } else {
            throw_unsup_format!("{open_name}: unsupported access mode {:#x}", access_mode);
        }

        if ecx.tcx.sess.target.os == Os::Linux {
            let o_largefile = ecx.eval_libc_i32("O_LARGEFILE");
            if flag & o_largefile == o_largefile {
                flag &= !o_largefile;
                // `fs::File` always opens in largefile mode so this is a NOP.
            }
            let o_tmpfile = ecx.eval_libc_i32("O_TMPFILE");
            // Note that this overaps with O_DIRECTORY! So we have to check O_TMPFILE first.
            if flag & o_tmpfile == o_tmpfile {
                // This flag requires a mode.
                read_mode(
                    ecx,
                    &mut options,
                    if dirfd.is_some() {
                        "openat(dirfd, pathname, O_TMPFILE, ...)"
                    } else {
                        "open(pathname, O_TMPFILE, ...)"
                    },
                )?;

                // We don't currently support this, error gracefully.
                return ecx.set_errno_and_return_neg1_i32(LibcError("EOPNOTSUPP"));
            }
        }
        let o_directory = ecx.eval_libc_i32("O_DIRECTORY");
        let mut want_directory = false;
        if flag & o_directory == o_directory {
            flag &= !o_directory;
            want_directory = true;
            // On Unix we can ask the host to only open directories. That's helpful especially in
            // combination with O_NOFOLLOW as it affects which error we get when the final component
            // is a symlink. Fixing the error up ourselves is non-trivial so we make the host
            // generate the right error.
            #[cfg(unix)]
            {
                custom_flags |= libc::O_DIRECTORY;
            }
        }
        let o_append = ecx.eval_libc_i32("O_APPEND");
        let append = flag & o_append == o_append;
        if append {
            flag &= !o_append;
            // The flag is a NOP if the file is not writable.
            if writable {
                options.append(true);
            }
        }
        let o_trunc = ecx.eval_libc_i32("O_TRUNC");
        if flag & o_trunc == o_trunc {
            flag &= !o_trunc;
            options.truncate(true);
        }
        let o_creat = ecx.eval_libc_i32("O_CREAT");
        if flag & o_creat == o_creat {
            flag &= !o_creat;
            // Get the mode.
            read_mode(
                ecx,
                &mut options,
                if dirfd.is_some() {
                    "openat(dirfd, pathname, O_CREAT, ...)"
                } else {
                    "open(pathname, O_CREAT, ...)"
                },
            )?;

            if want_directory {
                // O_CREAT + O_DIRECTORY is invalid.
                return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
            }

            // O_EXCL is only allowed if O_CREAT is set.
            let o_excl = ecx.eval_libc_i32("O_EXCL");
            if flag & o_excl == o_excl {
                flag &= !o_excl;
                options.create_new(true);
            } else {
                options.create(true);
            }
            // Creating a file requires opening it for writing.
            // Write accesses will still get rejected as `writable` is not set on the `FileHandle`.
            options.write(true);
        }
        let o_cloexec = ecx.eval_libc_i32("O_CLOEXEC");
        if flag & o_cloexec == o_cloexec {
            flag &= !o_cloexec;
            // We do not need to do anything for this flag because `std` already sets it.
            // (Technically we do not support *not* setting this flag, but we ignore that.)
        }
        let o_nofollow = ecx.eval_libc_i32("O_NOFOLLOW");
        let mut nofollow = false;
        if flag & o_nofollow == o_nofollow {
            flag &= !o_nofollow;
            nofollow = true;
            cfg_select! {
                unix => {
                    custom_flags |= libc::O_NOFOLLOW;
                }
                windows => {
                    if dirfd.is_some() {
                        throw_unsup_format!(
                            "openat: `O_NOFOLLOW` is not supported on Windows hosts"
                        );
                    }
                    custom_flags |=
                        windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
                }
            }
        }

        // If `flag` has any bits left set, those are not supported.
        if flag != 0 {
            throw_unsup_format!("{open_name}: unsupported flags: {flag:#x}");
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation(&format!("`{open_name}`"), reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
        }

        let dir = match ecx.get_dir(dirfd, &path) {
            Ok(dir) => dir,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        // Windows does not support "." relative to a `Dir` handle, but the empty path works.
        // (https://github.com/rust-lang/rust/issues/163032)
        let path = if cfg!(windows) && dirfd.is_some() && path.to_str() == Some(".") {
            Path::new("")
        } else {
            &path
        };

        // Let's see what we get when we open this!
        let f_or_d = open_file_or_dir(
            dir.as_ref().map(|d| &d.dir),
            path,
            options.custom_flags(custom_flags),
        );
        match f_or_d {
            Err(err) => ecx.set_errno_and_return_neg1_i32(err),
            Ok(Either::Right(dir)) => {
                // This means it cannot be a symlink, so `nofollow` is fine.
                if writable {
                    // On Windows, opening a folder writable can succeed.
                    // But here we want it to always fail.
                    return ecx.set_errno_and_return_neg1_i32(LibcError("EISDIR"));
                }

                let Some(fd) = ecx.machine.fds.insert_new(DirHandle { dir }) else {
                    return ecx.set_errno_and_return_neg1_i32(LibcError("EMFILE"));
                };
                interp_ok(Scalar::from_i32(fd))
            }
            Ok(Either::Left(file)) => {
                if want_directory {
                    return ecx.set_errno_and_return_neg1_i32(LibcError("ENOTDIR"));
                }
                if file.metadata().unwrap().is_symlink() {
                    if nofollow {
                        // On Windows, FILE_FLAG_OPEN_REPARSE_POINT makes opening still succeed, it
                        // just opens the symlink rather than the target. Turn that into an error.
                        return ecx.set_errno_and_return_neg1_i32(LibcError("ELOOP"));
                    }
                    panic!("we should not get a symlink here");
                }

                let Some(fd) =
                    ecx.machine.fds.insert_new(FileHandle { file, writable, readable, append })
                else {
                    return ecx.set_errno_and_return_neg1_i32(LibcError("EMFILE"));
                };
                interp_ok(Scalar::from_i32(fd))
            }
        }
    }

    fn lseek(
        &mut self,
        fd_num: i32,
        offset: i64,
        whence: i32,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        // Isolation check is done via `FileDescription` trait.

        let seek_from = if whence == ecx.eval_libc_i32("SEEK_SET") {
            let Ok(offset) = u64::try_from(offset) else {
                // Negative offsets return `EINVAL`.
                return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
            };
            SeekFrom::Start(offset)
        } else if whence == ecx.eval_libc_i32("SEEK_CUR") {
            SeekFrom::Current(offset)
        } else if whence == ecx.eval_libc_i32("SEEK_END") {
            SeekFrom::End(offset)
        } else {
            return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
        };

        let communicate = ecx.machine.communicate();

        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return ecx.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };
        let result = fd.seek(communicate, seek_from)?.map(|offset| i64::try_from(offset).unwrap());
        drop(fd);

        let result = ecx.try_unwrap_io_result(result)?;
        ecx.write_int(result, dest)?;
        interp_ok(())
    }

    fn unlink(&mut self, path_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let path = ecx.read_path_from_c_str(ecx.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`unlink`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
        }

        let result = host_remove_file(None, &path);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result.map(|()| 0))?))
    }

    fn unlinkat(
        &mut self,
        dirfd_op: &OpTy<'tcx>,
        path_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let dirfd = ecx.read_scalar(dirfd_op)?.to_i32()?;
        let path = ecx.read_path_from_c_str(ecx.read_pointer(path_op)?)?.into_owned();
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`unlinkat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        // We only support `AT_REMOVEDIR`, which makes this behave like `rmdir`.
        let remove_dir = if flags == 0 {
            false
        } else if flags == ecx.eval_libc_i32("AT_REMOVEDIR") {
            true
        } else {
            throw_unsup_format!("unsupported flags for `unlinkat`: {flags:#x}")
        };

        let dir = match ecx.get_dir(Some(dirfd), &path) {
            Ok(dir) => dir,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        let result = if remove_dir {
            host_remove_dir(dir.as_deref(), &path)
        } else {
            host_remove_file(dir.as_deref(), &path)
        };
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result.map(|()| 0))?))
    }

    fn symlink(
        &mut self,
        target_op: &OpTy<'tcx>,
        linkpath_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        fn create_link(target: &Path, link: &Path) -> std::io::Result<()> {
            cfg_select! {
                unix => std::os::unix::fs::symlink(target, link),
                windows => {
                    use std::os::windows::fs;

                    // The target filename is interpreted relative to where `link` is located,
                    // but we need it relative to where we are.
                    let target_is_dir = link.parent().is_some_and(|dir| dir.join(target).is_dir());
                    // This is racy, but not much we can do about that.
                    // FIXME: maybe we can retry based on the error code?
                    if target_is_dir {
                        fs::symlink_dir(target, link)
                    } else {
                        fs::symlink_file(target, link)
                    }
                }
            }
        }

        let ecx = self.eval_context_mut();
        let target = ecx.read_path_from_c_str(ecx.read_pointer(target_op)?)?;
        let linkpath = ecx.read_path_from_c_str(ecx.read_pointer(linkpath_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`symlink`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
        }

        let result = create_link(&target, &linkpath).map(|_| 0);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result)?))
    }

    fn linkat(
        &mut self,
        oldfd_op: &OpTy<'tcx>,
        oldpath_op: &OpTy<'tcx>,
        newfd_op: &OpTy<'tcx>,
        newpath_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        // Load all arguments
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;
        let oldfd = ecx.read_scalar(oldfd_op)?.to_i32()?;
        let newfd = ecx.read_scalar(newfd_op)?.to_i32()?;
        let oldpath_ptr = ecx.read_pointer(oldpath_op)?;
        let newpath_ptr = ecx.read_pointer(newpath_op)?;

        // Relevant libc constants
        let at_fdcwd = ecx.eval_libc_i32("AT_FDCWD");

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`linkat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
        }

        // Read flags - only support 0.
        if flags != 0 {
            throw_unsup_format!("unsupported linkat flags {:#x}", flags);
        }

        // Resolve oldpath
        if oldfd != at_fdcwd {
            throw_unsup_format!("linkat with `olddirfd` not equal to `AT_FDCWD` is not supported");
        }
        if ecx.ptr_is_null(oldpath_ptr)? {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }
        let oldpath = ecx.read_path_from_c_str(oldpath_ptr)?.into_owned();

        // Resolve newpath
        if newfd != at_fdcwd {
            throw_unsup_format!("linkat with `newdirfd` not equal to `AT_FDCWD` is not supported");
        }
        if ecx.ptr_is_null(newpath_ptr)? {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }
        let newpath = ecx.read_path_from_c_str(newpath_ptr)?.into_owned();

        let result = fs::hard_link(&oldpath, &newpath).map(|()| 0);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result)?))
    }

    fn stat(&mut self, path_op: &OpTy<'tcx>, buf_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        if !matches!(
            &ecx.tcx.sess.target.os,
            Os::MacOs | Os::FreeBsd | Os::Solaris | Os::Illumos | Os::Android | Os::Linux
        ) {
            panic!("`stat` should not be called on {}", ecx.tcx.sess.target.os);
        }

        let path_scalar = ecx.read_pointer(path_op)?;
        let path = ecx.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`stat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        let metadata = match FileMetadata::from_host(ecx, fs::metadata(path))? {
            Ok(metadata) => metadata,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };

        interp_ok(Scalar::from_i32(ecx.write_stat_buf(metadata, buf_op)?))
    }

    // `lstat` is used to get symlink metadata.
    fn lstat(&mut self, path_op: &OpTy<'tcx>, buf_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        if !matches!(
            &ecx.tcx.sess.target.os,
            Os::MacOs | Os::FreeBsd | Os::Solaris | Os::Illumos | Os::Android | Os::Linux
        ) {
            panic!("`lstat` should not be called on {}", ecx.tcx.sess.target.os);
        }

        let path_scalar = ecx.read_pointer(path_op)?;
        let path = ecx.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`lstat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        let metadata = match FileMetadata::from_host(ecx, fs::symlink_metadata(path))? {
            Ok(metadata) => metadata,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };

        interp_ok(Scalar::from_i32(ecx.write_stat_buf(metadata, buf_op)?))
    }

    fn fstat(&mut self, fd_op: &OpTy<'tcx>, buf_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        if !matches!(
            &ecx.tcx.sess.target.os,
            Os::MacOs | Os::FreeBsd | Os::Solaris | Os::Illumos | Os::Linux | Os::Android
        ) {
            panic!("`fstat` should not be called on {}", ecx.tcx.sess.target.os);
        }

        let fd = ecx.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`fstat`", reject_with)?;
            // Set error code as "EBADF" (bad fd); "EACCESS" does not make much sense here.
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        }

        let metadata = match FileMetadata::from_fd_num(ecx, fd)? {
            Ok(metadata) => metadata,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        interp_ok(Scalar::from_i32(ecx.write_stat_buf(metadata, buf_op)?))
    }

    fn fstatat(
        &mut self,
        dirfd_op: &OpTy<'tcx>,
        path_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        if !matches!(
            &ecx.tcx.sess.target.os,
            Os::MacOs | Os::FreeBsd | Os::Solaris | Os::Illumos | Os::Linux | Os::Android
        ) {
            panic!("`fstatat` should not be called on {}", ecx.tcx.sess.target.os);
        }

        let dirfd = ecx.read_scalar(dirfd_op)?.to_i32()?;
        let path = ecx.read_pointer(path_op)?;
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`fstatat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        // Parse flags.
        let mut flags = flags;
        // AT_SYMLINK_NOFOLLOW
        let at_symlink_nofollow = ecx.eval_libc_i32("AT_SYMLINK_NOFOLLOW");
        let symlink_nofollow_flag = flags & at_symlink_nofollow == at_symlink_nofollow;
        flags &= !at_symlink_nofollow;
        // Complain about unknown flags.
        if flags != 0 {
            throw_unsup_format!("unsupported flags for `fstatat`: {flags:#x}")
        }

        let path = ecx.read_path_from_c_str(path)?.into_owned();
        // Resolve dirfd + path to metadata.
        let metadata = FileMetadata::at(
            ecx,
            dirfd,
            &path,
            symlink_nofollow_flag,
            /* empty_path_flag */ false,
        )?;

        let metadata = match metadata {
            Ok(metadata) => metadata,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        interp_ok(Scalar::from_i32(ecx.write_stat_buf(metadata, buf_op)?))
    }

    fn linux_statx(
        &mut self,
        dirfd_op: &OpTy<'tcx>,    // Should be an `int`
        pathname_op: &OpTy<'tcx>, // Should be a `const char *`
        flags_op: &OpTy<'tcx>,    // Should be an `int`
        mask_op: &OpTy<'tcx>,     // Should be an `unsigned int`
        statxbuf_op: &OpTy<'tcx>, // Should be a `struct statx *`
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_linux_like("statx");

        let dirfd = ecx.read_scalar(dirfd_op)?.to_i32()?;
        let pathname_ptr = ecx.read_pointer(pathname_op)?;
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;
        let _mask = ecx.read_scalar(mask_op)?.to_u32()?;
        let statxbuf_ptr = ecx.read_pointer(statxbuf_op)?;

        // If the statxbuf or pathname pointers are null, the function fails with `EFAULT`.
        if ecx.ptr_is_null(statxbuf_ptr)? || ecx.ptr_is_null(pathname_ptr)? {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }

        let statxbuf = ecx.deref_pointer(statxbuf_op)?;
        let path = ecx.read_path_from_c_str(pathname_ptr)?.into_owned();

        // Parse flags.
        let mut flags = flags;
        // AT_EMPTY_PATH
        let at_empty_path = ecx.eval_libc_i32("AT_EMPTY_PATH");
        let empty_path_flag = flags & at_empty_path == at_empty_path;
        flags &= !at_empty_path;
        // AT_SYMLINK_NOFOLLOW
        let at_symlink_nofollow = ecx.eval_libc_i32("AT_SYMLINK_NOFOLLOW");
        let symlink_nofollow_flag = flags & at_symlink_nofollow == at_symlink_nofollow;
        flags &= !at_symlink_nofollow;
        // Complain about unknown flags.
        if flags != 0 {
            throw_unsup_format!("unsupported flags for `statx`: {flags:#x}")
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`statx`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        // Resolve dirfd + path to metadata.
        let metadata = FileMetadata::at(ecx, dirfd, &path, symlink_nofollow_flag, empty_path_flag)?;

        let metadata = match metadata {
            Ok(metadata) => metadata,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };

        // The `_mask_op` parameter specifies the file information that the caller requested.
        // However, `statx` is allowed to return information that was not requested or to not
        // return information that was requested. This `mask` represents the information we can
        // actually provide for any target.
        let mut mask = ecx.eval_libc_u32("STATX_TYPE")
            | ecx.eval_libc_u32("STATX_MODE")
            | ecx.eval_libc_u32("STATX_SIZE");

        // Check which pieces of metadata we acquired, and set the appropriate flags in the mask.
        if metadata.ino.is_some() {
            mask |= ecx.eval_libc_u32("STATX_INO");
        }
        if metadata.nlink.is_some() {
            mask |= ecx.eval_libc_u32("STATX_NLINK");
        }
        if metadata.uid.is_some() {
            mask |= ecx.eval_libc_u32("STATX_UID");
        }
        if metadata.gid.is_some() {
            mask |= ecx.eval_libc_u32("STATX_GID");
        }
        if metadata.blocks.is_some() {
            mask |= ecx.eval_libc_u32("STATX_BLOCKS");
        }

        // We need to set the corresponding bits of `mask` if the access, creation and modification
        // times were available. Otherwise we let them be zero.
        let (access_sec, access_nsec) = metadata
            .accessed
            .map(|tup| {
                mask |= ecx.eval_libc_u32("STATX_ATIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        let (created_sec, created_nsec) = metadata
            .created
            .map(|tup| {
                mask |= ecx.eval_libc_u32("STATX_BTIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        let (modified_sec, modified_nsec) = metadata
            .modified
            .map(|tup| {
                mask |= ecx.eval_libc_u32("STATX_MTIME");
                interp_ok(tup)
            })
            .unwrap_or_else(|| interp_ok((0, 0)))?;

        // Now we write everything to `statxbuf`. We write a zero for the unavailable fields.
        ecx.write_int_fields_named(
            &[
                ("stx_mask", mask.into()),
                ("stx_mode", metadata.mode.into()),
                ("stx_blksize", metadata.blksize.unwrap_or(0).into()),
                ("stx_nlink", metadata.nlink.unwrap_or(0).into()),
                ("stx_uid", metadata.uid.unwrap_or(0).into()),
                ("stx_gid", metadata.gid.unwrap_or(0).into()),
                ("stx_ino", metadata.ino.unwrap_or(0).into()),
                ("stx_size", metadata.size.into()),
                ("stx_blocks", metadata.blocks.unwrap_or(0).into()),
                // These fields do not seem to have a bit in `mask`, they are "always present",
                // so we make sure they get initialized.
                ("stx_attributes_mask", 0),
                ("stx_attributes", 0),
                ("stx_rdev_major", 0),
                ("stx_rdev_minor", 0),
                ("stx_dev_major", 0),
                ("stx_dev_minor", 0),
            ],
            &statxbuf,
        )?;
        #[rustfmt::skip]
        ecx.write_int_fields_named(
            &[
                ("tv_sec", access_sec.into()),
                ("tv_nsec", access_nsec.into()),
            ],
            &ecx.project_field_named(&statxbuf, "stx_atime")?,
        )?;
        #[rustfmt::skip]
        ecx.write_int_fields_named(
            &[
                ("tv_sec", created_sec.into()),
                ("tv_nsec", created_nsec.into()),
            ],
            &ecx.project_field_named(&statxbuf, "stx_btime")?,
        )?;
        #[rustfmt::skip]
        ecx.write_int_fields_named(
            &[
                ("tv_sec", 0.into()),
                ("tv_nsec", 0.into()),
            ],
            &ecx.project_field_named(&statxbuf, "stx_ctime")?,
        )?;
        #[rustfmt::skip]
        ecx.write_int_fields_named(
            &[
                ("tv_sec", modified_sec.into()),
                ("tv_nsec", modified_nsec.into()),
            ],
            &ecx.project_field_named(&statxbuf, "stx_mtime")?,
        )?;

        // The remaining fields have a mask bit associated with them, a bit that we do not
        // set. Therefore we also do not initialize them.

        interp_ok(Scalar::from_i32(0))
    }

    fn chmod(
        &mut self,
        dirfd_op: Option<&OpTy<'tcx>>,
        path_op: &OpTy<'tcx>,
        mode_op: &OpTy<'tcx>,
        flags_op: Option<&OpTy<'tcx>>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        let operation_name = if dirfd_op.is_some() { "`fchmodat`" } else { "`chmod`" };

        let dirfd = dirfd_op.try_map(|dirfd_op| ecx.read_scalar(dirfd_op)?.to_i32())?;
        let path_ptr = ecx.read_pointer(path_op)?;
        let mode = ecx.read_int::<u32>(mode_op)?;
        let flags =
            if let Some(flags_op) = flags_op { ecx.read_scalar(flags_op)?.to_i32()? } else { 0 };

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation(operation_name, reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        // Get path and directory.
        if ecx.ptr_is_null(path_ptr)? {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }
        let path = ecx.read_path_from_c_str(path_ptr)?;
        let dir = match ecx.get_dir(dirfd, &path) {
            Ok(dir) => dir,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        if dir.is_some() {
            throw_unsup_format!("{operation_name} with a directory handle is not supported");
        }

        // Parse flags.
        let mut flags = flags;
        // AT_SYMLINK_NOFOLLOW
        let at_symlink_nofollow = ecx.eval_libc_i32("AT_SYMLINK_NOFOLLOW");
        let nofollow = flags & at_symlink_nofollow == at_symlink_nofollow;
        flags &= !at_symlink_nofollow;
        // Complain about unknown flags.
        if flags != 0 {
            throw_unsup_format!("unsupported flags for `fchmodat`: {flags:#x}")
        }

        // Perform actual permission update.
        let permissions = ecx.host_permissions_from_mode(mode, || {
            // We get here if we need to read the old permissions before we can set new ones.
            // This is fundamentally racy. We cannot even use a file handle to make it less racy:
            // setting permissions requires opening for writing, but opening for writing fails
            // if the file is not already writable... so how do we make a read-only file writable?
            if nofollow { fs::symlink_metadata(&path) } else { fs::metadata(&path) }
                .map(|m| m.permissions())
        })?;
        let res = permissions.and_then(|permissions| {
            if nofollow {
                fs::set_permissions_nofollow(path, permissions)
            } else {
                fs::set_permissions(path, permissions)
            }
        });
        if let Err(err) = res {
            return ecx.set_errno_and_return_neg1_i32(err);
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn fchmod(&mut self, fd_op: &OpTy<'tcx>, mode_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let fd_num = ecx.read_scalar(fd_op)?.to_i32()?;
        let mode = ecx.read_int::<u32>(mode_op)?;

        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        let Some(file) = fd.downcast::<FileHandle>() else {
            // The docs don't talk about what happens for non-regular files...
            throw_unsup_format!("`fchmod` is only supported on regular files")
        };
        if !file.writable && !file.readable {
            // Apparently, `fchmod` on a read-only file is fine. But let's not allow it on a
            // path-only file.
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        }
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        let permissions = ecx.host_permissions_from_mode(mode, || {
            // We get here if we need to read the old permissions before we can set new ones.
            // This is fundamentally racy.
            file.file.metadata().map(|m| m.permissions())
        })?;
        let res = permissions.and_then(|permissions| file.file.set_permissions(permissions));
        if let Err(err) = res {
            return ecx.set_errno_and_return_neg1_i32(err);
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn rename(
        &mut self,
        oldfd: Option<&OpTy<'tcx>>,
        oldpath_op: &OpTy<'tcx>,
        newfd: Option<&OpTy<'tcx>>,
        newpath_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let oldfd = oldfd.try_map(|oldfd| ecx.read_scalar(oldfd)?.to_i32())?;
        let oldpath_ptr = ecx.read_pointer(oldpath_op)?;
        let newfd = newfd.try_map(|newfd| ecx.read_scalar(newfd)?.to_i32())?;
        let newpath_ptr = ecx.read_pointer(newpath_op)?;

        if ecx.ptr_is_null(oldpath_ptr)? || ecx.ptr_is_null(newpath_ptr)? {
            return ecx.set_errno_and_return_neg1(LibcError("EFAULT"), dest);
        }

        let oldpath = ecx.read_path_from_c_str(oldpath_ptr)?;
        let newpath = ecx.read_path_from_c_str(newpath_ptr)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            let name = if oldfd.is_some() || newfd.is_some() { "`renameat`" } else { "`rename`" };
            ecx.reject_in_isolation(name, reject_with)?;
            return ecx.set_errno_and_return_neg1(ErrorKind::PermissionDenied, dest);
        }

        let old_dir = ecx.get_dir(oldfd, &oldpath);
        let new_dir = ecx.get_dir(newfd, &newpath);

        let result = match (old_dir, new_dir) {
            (Ok(None), Ok(None)) => fs::rename(&oldpath, &newpath),
            (Ok(Some(old)), Ok(Some(new))) => old.dir.rename(&oldpath, &new.dir, &newpath),
            (Ok(_), Ok(_)) =>
                throw_unsup_format!("`renameat` with only one path relative to a directory fd"),
            (Err(err), _) | (_, Err(err)) => return ecx.set_errno_and_return_neg1(err, dest),
        };
        let result = ecx.try_unwrap_io_result(result.map(|()| 0i32))?;
        ecx.write_int(result, dest)
    }

    fn mkdir(
        &mut self,
        dirfd_op: Option<&OpTy<'tcx>>,
        path_op: &OpTy<'tcx>,
        mode_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        let mkdir_name = if dirfd_op.is_some() { "mkdirat" } else { "mkdir" };

        let dirfd = dirfd_op.try_map(|dirfd_op| ecx.read_scalar(dirfd_op)?.to_i32())?;
        let path = ecx.read_path_from_c_str(ecx.read_pointer(path_op)?)?;
        let mode = ecx.read_int::<u32>(mode_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation(&format!("`{mkdir_name}`"), reject_with)?;
            return ecx.set_errno_and_return_neg1(LibcError("EACCES"), dest);
        }

        let dir = match ecx.get_dir(dirfd, &path) {
            Ok(dir) => dir,
            Err(err) => return ecx.set_errno_and_return_neg1(err, dest),
        };

        let result = match dir {
            Some(dir) => {
                // `Dir::create_dir` does not take a mode; std always uses 0o777 internally.
                // So we can only support that mode here.
                if mode != 0o777 {
                    throw_unsup_format!(
                        "`mkdirat` relative to a directory file descriptor with mode {mode:#o} is not supported \
                        (only 0o777 is supported)"
                    );
                }
                dir.dir.create_dir(path)
            }
            None => {
                #[cfg_attr(not(unix), allow(unused_mut))]
                let mut builder = DirBuilder::new();

                // On Unix hosts, forward the mode of the directory
                // (i.e. permission bits and the sticky bit).
                // Other hosts can't represent it, so we only support the default mode there.
                cfg_select! {
                    unix => {
                        use std::os::unix::fs::DirBuilderExt;
                        builder.mode(mode);
                    }
                    _ =>
                        if mode != 0o777 {
                            throw_unsup_format!(
                                "`{mkdir_name}` with mode {mode:#o} is not supported on non-Unix hosts \
                                (only 0o777 is supported)"
                            );
                        },
                }

                builder.create(path)
            }
        };

        ecx.return_io_result_via_errno(result.map(|()| 0), dest)?;
        interp_ok(())
    }

    fn rmdir(&mut self, path_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let path = ecx.read_path_from_c_str(ecx.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`rmdir`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(ErrorKind::PermissionDenied);
        }

        let result = host_remove_dir(None, &path);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result.map(|_| 0i32))?))
    }

    fn opendir(&mut self, name_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let name = ecx.read_path_from_c_str(ecx.read_pointer(name_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`opendir`", reject_with)?;
            ecx.set_last_error(LibcError("EACCES"))?;
            return interp_ok(Scalar::null_ptr(ecx));
        }

        let result = fs::read_dir(&name);

        match result {
            Ok(read_dir) => {
                // Create an underlying FD as well, in case someone calls `dirfd` later.
                // (We could do this lazily but that does not seem worth it.)
                let Ok(dir) = read_dir.dir() else {
                    throw_unsup_format!(
                        "cannot `opendir` this directory: failed to create directory handle"
                    );
                };
                let dir = ecx.machine.fds.new_ref(DirHandle { dir });
                let dir_fd_id = dir.id();
                let Some(dir_fd_num) = ecx.machine.fds.insert(dir) else {
                    return ecx.set_errno_and_return_neg1_i32(LibcError("EMFILE"));
                };

                let stream = DirStream::new(read_dir, dir_fd_num, dir_fd_id);
                let id = ecx.machine.dirs.insert_new(stream);

                // The libc API for opendir says that this method returns a pointer to an opaque
                // structure, but we are returning an ID number. Thus, pass it as a scalar of
                // pointer width. This also conveniently means that any attempt to access
                // memory via this pointer raises UB.
                interp_ok(Scalar::from_target_usize(id, ecx))
            }
            Err(e) => {
                ecx.set_last_error(e)?;
                interp_ok(Scalar::null_ptr(ecx))
            }
        }
    }

    fn readdir(
        &mut self,
        dirp_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
        buf_type: &'static str,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        if !matches!(
            &ecx.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::Solaris | Os::Illumos | Os::FreeBsd | Os::MacOs
        ) {
            throw_unsup_format!("`readdir` is not yet supported on {}", ecx.tcx.sess.target.os);
        }

        let dirp = ecx.read_target_usize(dirp_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`readdir`", reject_with)?;
            ecx.set_last_error(LibcError("EBADF"))?;
            ecx.write_int(0, dest)?;
            return interp_ok(());
        }

        let open_dir = ecx.machine.dirs.streams.get_mut(&dirp).ok_or_else(|| {
            err_ub_format!("the DIR pointer passed to `readdir` did not come from opendir")
        })?;

        // Check that the backing FD is still valid.
        if ecx.machine.fds.get(open_dir.fd_num).is_none_or(|fd| fd.id() != open_dir.fd_id) {
            throw_ub_format!("the backing FD for this DIR stream has been tampered with");
        }

        let entry = match open_dir.next_host_entry() {
            Some(Ok(dir_entry)) => {
                let dir_entry = ecx.dir_entry_fields(dir_entry)?;

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
                //
                // On FreeBSD:
                // pub struct dirent {
                //     pub d_fileno: uint32_t,
                //     pub d_reclen: uint16_t,
                //     pub d_type: uint8_t,
                //     pub d_namlen: uint8_t,
                //     pub d_name: [c_char; 256],
                // }
                //
                // On macOS:
                // pub struct dirent {
                //     pub d_ino: u64,
                //     pub d_seekoff: u64,
                //     pub d_reclen: u16,
                //     pub d_namlen: u16,
                //     pub d_type: u8,
                //     pub d_name: [c_char; 1024],
                // }

                let dirent_layout = ecx.libc_ty_layout(buf_type);
                let fields = &dirent_layout.fields;
                let d_name_offset = fields.offset(fields.count().strict_sub(1)).bytes();

                // Determine the size of the buffer we have to allocate.
                let mut name = dir_entry.name; // not a Path as there are no separators!
                name.push("\0"); // Add a NUL terminator
                let name_bytes = name.as_encoded_bytes();
                let name_len = u64::try_from(name_bytes.len()).unwrap();
                let size = d_name_offset.strict_add(name_len);

                let entry = ecx.allocate_ptr(
                    Size::from_bytes(size),
                    dirent_layout.align,
                    MiriMemoryKind::Runtime.into(),
                    AllocInit::Uninit,
                )?;
                let entry = ecx.ptr_to_mplace(entry.into(), dirent_layout);

                // Write the name.
                // The name is not a normal field, we already computed the offset above.
                let name_ptr = entry.ptr().wrapping_offset(Size::from_bytes(d_name_offset), ecx);
                ecx.write_bytes_ptr(name_ptr, name_bytes.iter().copied())?;

                // Write common fields.
                let ino_name =
                    if ecx.tcx.sess.target.os == Os::FreeBsd { "d_fileno" } else { "d_ino" };
                ecx.write_int_fields_named(
                    &[(ino_name, dir_entry.ino.into()), ("d_reclen", size.into())],
                    &entry,
                )?;

                // Write "optional" fields.
                if let Some(d_off) = ecx.try_project_field_named(&entry, "d_off")? {
                    ecx.write_int(0, &d_off)?;
                }
                if let Some(d_seekoff) = ecx.try_project_field_named(&entry, "d_seekoff")? {
                    ecx.write_int(0, &d_seekoff)?;
                }
                if let Some(d_namlen) = ecx.try_project_field_named(&entry, "d_namlen")? {
                    ecx.write_int(name_len.strict_sub(1), &d_namlen)?;
                }
                if let Some(d_type) = ecx.try_project_field_named(&entry, "d_type")? {
                    ecx.write_int(dir_entry.d_type, &d_type)?;
                }

                Some(entry.ptr())
            }
            None => {
                // end of stream: return NULL
                None
            }
            Some(Err(e)) => {
                ecx.set_last_error(e)?;
                None
            }
        };

        let open_dir = ecx.machine.dirs.streams.get_mut(&dirp).unwrap();
        if let Some((old_entry, old_type)) =
            std::mem::replace(&mut open_dir.entry, entry.map(|e| (e, buf_type)))
        {
            if old_type != buf_type {
                throw_ub_format!("mixing different `readdir` variants on the same `DIR` instance");
            }
            ecx.deallocate_ptr(old_entry, None, MiriMemoryKind::Runtime.into())?;
        }

        ecx.write_pointer(entry.unwrap_or_else(Pointer::null), dest)?;
        interp_ok(())
    }

    fn dirfd(&mut self, dirp_op: &OpTy<'tcx>, dest: &MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let dirp = ecx.read_target_usize(dirp_op)?;

        let open_dir = ecx.machine.dirs.streams.get_mut(&dirp).ok_or_else(|| {
            err_ub_format!("the DIR pointer passed to `readdir` did not come from opendir")
        })?;

        // Check that the backing FD is still valid.
        if ecx.machine.fds.get(open_dir.fd_num).is_none_or(|fd| fd.id() != open_dir.fd_id) {
            throw_ub_format!("the backing FD for this DIR stream has been tampered with");
        }

        let fd_num = open_dir.fd_num;
        ecx.write_int(fd_num, dest)?;
        interp_ok(())
    }

    fn closedir(&mut self, dirp_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let dirp = ecx.read_target_usize(dirp_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`closedir`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        }

        let mut open_dir = ecx.machine.dirs.streams.remove(&dirp).ok_or_else(|| {
            err_ub_format!("the DIR pointer passed to `closedir` did not come from opendir")
        })?;

        // Check that the backing FD is still valid.
        if ecx.machine.fds.get(open_dir.fd_num).is_none_or(|fd| fd.id() != open_dir.fd_id) {
            throw_ub_format!("the backing FD for this DIR stream has been tampered with");
        }
        // And close it.
        ecx.close(open_dir.fd_num)?;

        if let Some((entry, _type)) = open_dir.entry.take() {
            ecx.deallocate_ptr(entry, None, MiriMemoryKind::Runtime.into())?;
        }
        // We drop the `open_dir`, which will close the host dir handle.
        drop(open_dir);

        interp_ok(Scalar::from_i32(0))
    }

    fn ftruncate64(&mut self, fd_num: i32, length: i64) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        let Some(file) = fd.downcast::<FileHandle>() else {
            // The docs say that EINVAL is returned when the FD "does not reference a regular file
            // or a POSIX shared memory object" (and we don't support shmem objects).
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };
        if !file.writable {
            // man page says "EBADF or EINVAL", Linux seems to use EINVAL.
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        if let Ok(length) = length.try_into() {
            let result = file.file.set_len(length);
            let result = ecx.try_unwrap_io_result(result.map(|_| 0i32))?;
            interp_ok(Scalar::from_i32(result))
        } else {
            ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"))
        }
    }

    /// NOTE: According to the man page of `possix_fallocate`, it returns the error code instead
    /// of setting `errno`.
    fn posix_fallocate(
        &mut self,
        fd_num: i32,
        offset: i64,
        len: i64,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`posix_fallocate`", reject_with)?;
            // Return error code "EBADF" (bad fd).
            return interp_ok(ecx.eval_libc("EBADF"));
        }

        match ecx.fallocate_impl(fd_num, offset, len)? {
            Ok(()) => interp_ok(Scalar::from_i32(0)),
            Err(e) => ecx.io_error_to_errnum(e),
        }
    }

    fn linux_fallocate(
        &mut self,
        fd: i32,
        mode: i32,
        offset: i64,
        size: i64,
    ) -> InterpResult<'tcx, Scalar> {
        // This is mostly a copy of `posix_fallocate` except that errors are returned via errno.
        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_linux_like("fallocate");

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`fallocate`", reject_with)?;
            // Set error code "EBADF" (bad fd).
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        }

        // We only support `fallocate` as a replacement for `posix_fallocate` on linux,
        // so a non-default `mode` is not supported.
        if mode != 0 {
            throw_unsup_format!("unsupported flags for `fallocate` in `mode` argument: {mode:#x}")
        }

        match ecx.fallocate_impl(fd, offset, size)? {
            Ok(()) => interp_ok(Scalar::from_i32(0)),
            Err(e) => ecx.set_errno_and_return_neg1_i32(e),
        }
    }

    /// Shared logic between `posix_fallocate` and `linux_fallocate`.
    fn fallocate_impl(
        &self,
        fd_num: i32,
        offset: i64,
        len: i64,
    ) -> InterpResult<'tcx, Result<(), IoError>> {
        let ecx = self.eval_context_ref();

        // EINVAL is returned/set when: "offset was less than 0, or len was less than or equal to 0".
        if offset < 0 || len <= 0 {
            return interp_ok(Err(LibcError("EINVAL")));
        }

        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return interp_ok(Err(LibcError("EBADF")));
        };
        let Some(file) = fd.downcast::<FileHandle>() else {
            // Man page specifies to return ENODEV if `fd` is not a regular file.
            return interp_ok(Err(LibcError("ENODEV")));
        };

        if !file.writable {
            return interp_ok(Err(LibcError("EBADF")));
        }

        let current_size = match file.file.metadata() {
            Ok(metadata) => metadata.len(),
            Err(err) => return interp_ok(Err(err.into())),
        };

        // Checked i64 addition, to ensure the result does not exceed the max file size.
        let new_size = match offset.checked_add(len) {
            // `new_size` is definitely non-negative, so we can cast to `u64`.
            Some(new_size) => u64::try_from(new_size).unwrap(),
            None => return interp_ok(Err(LibcError("EFBIG"))), // new size too big
        };

        // If the size of the file is less than offset+size, then the file is increased to this
        // size; otherwise the file size is left unchanged.
        if current_size < new_size {
            match file.file.set_len(new_size) {
                Ok(()) => interp_ok(Ok(())),
                Err(err) => interp_ok(Err(err.into())),
            }
        } else {
            interp_ok(Ok(()))
        }
    }

    fn fsync(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        // On macOS, `fsync` (unlike `fcntl(F_FULLFSYNC)`) does not wait for the
        // underlying disk to finish writing. In the interest of host compatibility,
        // we conservatively implement this with `sync_all`, which
        // *does* wait for the disk.

        let ecx = self.eval_context_mut();

        let fd = ecx.read_scalar(fd_op)?.to_i32()?;

        ecx.ffullsync_fd(fd)
    }

    fn ffullsync_fd(&mut self, fd_num: i32) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`fsync` is only supported on file-backed file descriptors")
        })?;
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        let io_result = maybe_sync_file(&file, File::sync_all);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(io_result)?))
    }

    fn fdatasync(&mut self, fd_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let fd = ecx.read_scalar(fd_op)?.to_i32()?;

        let Some(fd) = ecx.machine.fds.get(fd) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`fdatasync` is only supported on file-backed file descriptors")
        })?;
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        let io_result = maybe_sync_file(&file, File::sync_data);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(io_result)?))
    }

    /// `futimens(fd, times)`: set `fd`'s access/modification times. `times` is `[atime, mtime]`, or
    /// NULL to set both to now.
    fn futimens(
        &mut self,
        fd_op: &OpTy<'tcx>,
        times_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let fd_num = ecx.read_scalar(fd_op)?.to_i32()?;
        let times = ecx.deref_pointer(times_op)?;

        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`futimens` is only supported on file-backed file descriptors")
        })?;
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        let Some((access, modified)) = ecx.parse_utimens_times(&times)? else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };
        let result = file.file.set_times(host_file_times_from_updates(access, modified));
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result.map(|()| 0i32))?))
    }

    /// `utimensat(dirfd, path, times, flags)`: like `futimens`, but for the file at `path`. With
    /// `AT_SYMLINK_NOFOLLOW`, a symlink at `path` is updated itself rather than its target.
    fn utimensat(
        &mut self,
        dirfd_op: &OpTy<'tcx>,
        path_op: &OpTy<'tcx>,
        times_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let dirfd = ecx.read_scalar(dirfd_op)?.to_i32()?;
        let path_ptr = ecx.read_pointer(path_op)?;
        let times = ecx.deref_pointer(times_op)?;
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;

        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`utimensat`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
        }

        if ecx.ptr_is_null(path_ptr)? {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }
        let path = ecx.read_path_from_c_str(path_ptr)?.into_owned();
        let dir = match ecx.get_dir(Some(dirfd), &path) {
            Ok(dir) => dir,
            Err(err) => return ecx.set_errno_and_return_neg1_i32(err),
        };
        if dir.is_some() {
            throw_unsup_format!("`utimensat` with a directory handle is not supported");
        }

        let mut flags = flags;
        // Parse known flags and subtract them from `flags`.
        let at_symlink_nofollow = ecx.eval_libc_i32("AT_SYMLINK_NOFOLLOW");
        let symlink_nofollow_flag = flags & at_symlink_nofollow == at_symlink_nofollow;
        flags &= !at_symlink_nofollow;

        if flags != 0 {
            throw_unsup_format!("unsupported flags for `utimensat`: {flags:#x}")
        }

        let Some((access, modified)) = ecx.parse_utimens_times(&times)? else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };
        if matches!((access, modified), (TimeUpdate::Omit, TimeUpdate::Omit)) {
            // Nothing to do. Behavior in this case differs between platforms: Linux always
            // succeeds, macOS/FreeBSD check whether the file exists. POSIX arguably allows both.
            // We decide to follow Linux.
            return interp_ok(Scalar::from_i32(0));
        }
        let filetimes = host_file_times_from_updates(access, modified);
        let result = if symlink_nofollow_flag {
            fs::set_times_nofollow(path, filetimes)
        } else {
            fs::set_times(path, filetimes)
        };
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(result.map(|()| 0i32))?))
    }

    fn linux_sync_file_range(
        &mut self,
        fd_op: &OpTy<'tcx>,
        offset_op: &OpTy<'tcx>,
        nbytes_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_linux_like("sync_file_range");

        let fd = ecx.read_scalar(fd_op)?.to_i32()?;
        let offset = ecx.read_scalar(offset_op)?.to_i64()?;
        let nbytes = ecx.read_scalar(nbytes_op)?.to_i64()?;
        let flags = ecx.read_scalar(flags_op)?.to_i32()?;

        if offset < 0 || nbytes < 0 {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }
        let allowed_flags = ecx.eval_libc_i32("SYNC_FILE_RANGE_WAIT_BEFORE")
            | ecx.eval_libc_i32("SYNC_FILE_RANGE_WRITE")
            | ecx.eval_libc_i32("SYNC_FILE_RANGE_WAIT_AFTER");
        if flags & allowed_flags != flags {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        }

        let Some(fd) = ecx.machine.fds.get(fd) else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        // Only regular files support synchronization.
        let file = fd.downcast::<FileHandle>().ok_or_else(|| {
            err_unsup_format!("`sync_data_range` is only supported on file-backed file descriptors")
        })?;
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        let io_result = maybe_sync_file(&file, File::sync_data);
        interp_ok(Scalar::from_i32(ecx.try_unwrap_io_result(io_result)?))
    }

    fn readlink(
        &mut self,
        pathname_op: &OpTy<'tcx>,
        buf_op: &OpTy<'tcx>,
        bufsize_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, i64> {
        let ecx = self.eval_context_mut();

        let pathname = ecx.read_os_str_from_c_str(ecx.read_pointer(pathname_op)?)?;
        let buf = ecx.read_pointer(buf_op)?;
        let bufsize = ecx.read_target_usize(bufsize_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`readlink`", reject_with)?;
            ecx.set_last_error(LibcError("EACCES"))?;
            return interp_ok(-1);
        }

        // Special case the procfs link to the current executable so that
        // `std::env::current_exe` works in cross-execution.
        let self_exe = match ecx.tcx.sess.target.os {
            Os::Linux | Os::Android => Some("/proc/self/exe"),
            Os::Solaris | Os::Illumos => Some("/proc/self/path/a.out"),
            _ => None,
        };
        let result = if self_exe.is_some() && pathname.to_str() == self_exe {
            ecx.machine.current_exe.clone().ok_or(ErrorKind::NotFound.into())
        } else if matches!(
            ecx.tcx.sess.target.os,
            Os::Linux | Os::Android | Os::Illumos | Os::Solaris
        ) && path::absolute(pathname).is_ok_and(|path| path.starts_with("/proc"))
        {
            // Trying to read a symlink inside `/proc` is likely going to be nonsense, e.g.
            // `/proc/self/fd/N` doesn't use the same FD numbering. So we pretend nothing exists.
            ecx.set_last_error(ErrorKind::NotFound)?;
            return interp_ok(-1);
        } else {
            // We read `pathname` as `OsStr` above so we could do the /proc/self/exe check.
            // But now we need a (host) path.
            let pathname = ecx.convert_path(Cow::Borrowed(pathname), PathConversion::TargetToHost);
            std::fs::read_link(pathname)
        };

        match result {
            Ok(resolved) => {
                // 'readlink' truncates the resolved path if the provided buffer is not large
                // enough, and does *not* add a null terminator. That means we cannot use the usual
                // `write_path_to_c_str` and have to re-implement parts of it ourselves.
                let resolved = ecx
                    .convert_path(Cow::Borrowed(resolved.as_ref()), PathConversion::HostToTarget);
                let mut path_bytes = resolved.as_encoded_bytes();
                let bufsize: usize = bufsize.try_into().unwrap();
                if path_bytes.len() > bufsize {
                    path_bytes = &path_bytes[..bufsize]
                }
                ecx.write_bytes_ptr(buf, path_bytes.iter().copied())?;
                interp_ok(path_bytes.len().try_into().unwrap())
            }
            Err(e) => {
                ecx.set_last_error(e)?;
                interp_ok(-1)
            }
        }
    }

    fn isatty(&mut self, miri_fd: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        // "returns 1 if fd is an open file descriptor referring to a terminal;
        // otherwise 0 is returned, and errno is set to indicate the error"
        let fd = ecx.read_scalar(miri_fd)?.to_i32()?;
        let error = if let Some(fd) = ecx.machine.fds.get(fd) {
            if fd.is_tty(ecx.machine.communicate()) {
                return interp_ok(Scalar::from_i32(1));
            } else {
                LibcError("ENOTTY")
            }
        } else {
            // FD does not exist
            LibcError("EBADF")
        };
        ecx.set_last_error(error)?;
        interp_ok(Scalar::from_i32(0))
    }

    fn realpath(
        &mut self,
        path_op: &OpTy<'tcx>,
        processed_path_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_unix("realpath");

        let pathname = ecx.read_path_from_c_str(ecx.read_pointer(path_op)?)?;
        let processed_ptr = ecx.read_pointer(processed_path_op)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`realpath`", reject_with)?;
            ecx.set_last_error(LibcError("EACCES"))?;
            return interp_ok(Scalar::from_target_usize(0, ecx));
        }

        let result = std::fs::canonicalize(pathname);
        match result {
            Ok(resolved) => {
                let path_max =
                    ecx.eval_libc_i32("PATH_MAX").try_into().expect("PATH_MAX does not fit in u64");
                let dest = if ecx.ptr_is_null(processed_ptr)? {
                    // POSIX says behavior when passing a null pointer is implementation-defined,
                    // but GNU/linux, freebsd, netbsd, bionic/android, and macos all treat a null pointer
                    // similarly to:
                    //
                    // "If resolved_path is specified as NULL, then realpath() uses
                    // malloc(3) to allocate a buffer of up to PATH_MAX bytes to hold
                    // the resolved pathname, and returns a pointer to this buffer.  The
                    // caller should deallocate this buffer using free(3)."
                    // <https://man7.org/linux/man-pages/man3/realpath.3.html>
                    ecx.alloc_path_as_c_str(&resolved, MiriMemoryKind::C.into())?
                } else {
                    let (wrote_path, _) =
                        ecx.write_path_to_c_str(&resolved, processed_ptr, path_max)?;

                    if !wrote_path {
                        // Note that we do not explicitly handle `FILENAME_MAX`
                        // (different from `PATH_MAX` above) as it is Linux-specific and
                        // seems like a bit of a mess anyway: <https://eklitzke.org/path-max-is-tricky>.
                        ecx.set_last_error(LibcError("ENAMETOOLONG"))?;
                        return interp_ok(Scalar::from_target_usize(0, ecx));
                    }
                    processed_ptr
                };

                interp_ok(Scalar::from_maybe_pointer(dest, ecx))
            }
            Err(e) => {
                ecx.set_last_error(e)?;
                interp_ok(Scalar::from_target_usize(0, ecx))
            }
        }
    }
    fn mkstemp(&mut self, template_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        use rand::seq::IndexedRandom;

        // POSIX defines the template string.
        const TEMPFILE_TEMPLATE_STR: &str = "XXXXXX";

        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_unix("mkstemp");

        // POSIX defines the maximum number of attempts before failure.
        //
        // `mkstemp()` relies on `tmpnam()` which in turn relies on `TMP_MAX`.
        // POSIX says this about `TMP_MAX`:
        // * Minimum number of unique filenames generated by `tmpnam()`.
        // * Maximum number of times an application can call `tmpnam()` reliably.
        //   * The value of `TMP_MAX` is at least 25.
        //   * On XSI-conformant systems, the value of `TMP_MAX` is at least 10000.
        // See <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/stdio.h.html>.
        let max_attempts = ecx.eval_libc_u32("TMP_MAX");

        // Get the raw bytes from the template -- as a byte slice, this is a string in the target
        // (and the target is unix, so a byte slice is the right representation).
        let template_ptr = ecx.read_pointer(template_op)?;
        let mut template = ecx.eval_context_ref().read_c_str(template_ptr)?.to_owned();
        let template_bytes = template.as_mut_slice();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = ecx.machine.isolated_op {
            ecx.reject_in_isolation("`mkstemp`", reject_with)?;
            return ecx.set_errno_and_return_neg1_i32(LibcError("EACCES"));
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
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
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

        cfg_select! {
            unix => {
                use std::os::unix::fs::OpenOptionsExt;
                // Do not allow others to read or modify this file.
                fopts.mode(0o600);
                fopts.custom_flags(libc::O_EXCL);
            }
            windows => {
                use std::os::windows::fs::OpenOptionsExt;
                // Do not allow others to read or modify this file.
                fopts.share_mode(0);
            }
            _ => {
                throw_unsup_format!("`mkstemp` is not supported on this host OS");
            }
        }

        // If the generated file already exists, we will try again `max_attempts` many times.
        for _ in 0..max_attempts {
            let rng = ecx.machine.rng.get_mut();

            // Generate a random unique suffix.
            let unique_suffix =
                (0..6).map(|_| SUBSTITUTIONS.choose(rng).unwrap()).collect::<String>();

            // Replace the template string with the random string.
            template_bytes[start_pos..end_pos].copy_from_slice(unique_suffix.as_bytes());

            // Write the modified template back to the passed in pointer to maintain POSIX semantics.
            ecx.write_bytes_ptr(template_ptr, template_bytes.iter().copied())?;

            // See if we can create and open this file.
            let file = fopts.open(bytes_to_os_str(template_bytes)?);
            match file {
                Ok(f) => {
                    let Some(fd) = ecx.machine.fds.insert_new(FileHandle {
                        file: f,
                        writable: true,
                        readable: true,
                        append: false,
                    }) else {
                        return ecx.set_errno_and_return_neg1_i32(LibcError("EMFILE"));
                    };
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
                            return ecx.set_errno_and_return_neg1_i32(e);
                        }
                    },
            }
        }

        // We ran out of attempts to create the file, return an error.
        ecx.set_errno_and_return_neg1_i32(LibcError("EEXIST"))
    }

    /// Only supports null offset pointers, i.e. copying from and advancing the file positions.
    fn linux_copy_file_range(
        &mut self,
        fd_in_op: &OpTy<'tcx>,
        off_in_op: &OpTy<'tcx>,
        fd_out_op: &OpTy<'tcx>,
        off_out_op: &OpTy<'tcx>,
        len_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        ecx.assert_target_os_is_linux_like("copy_file_range");

        let fd_in = ecx.read_scalar(fd_in_op)?.to_i32()?;
        let off_in = ecx.read_pointer(off_in_op)?;
        let fd_out = ecx.read_scalar(fd_out_op)?.to_i32()?;
        let off_out = ecx.read_pointer(off_out_op)?;
        let len = ecx.read_target_usize(len_op)?;
        let flags = ecx.read_scalar(flags_op)?.to_u32()?;

        let (Some(fd_in), Some(fd_out)) = (ecx.machine.fds.get(fd_in), ecx.machine.fds.get(fd_out))
        else {
            return ecx.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };
        if flags != 0 {
            throw_unsup_format!("`copy_file_range` with non-zero flags is not supported");
        }
        if !ecx.ptr_is_null(off_in)? || !ecx.ptr_is_null(off_out)? {
            throw_unsup_format!("`copy_file_range` with non-null offset pointers is not supported");
        }
        // Like Linux, check for directories (EISDIR), then other non-regular files (EINVAL),
        // then the access modes (EBADF).
        let is_dir = |fd: &DynFileDescriptionRef| fd.clone().downcast::<DirHandle>().is_some();
        let as_regular_file = |fd: DynFileDescriptionRef| {
            let file = fd.downcast::<FileHandle>()?;
            // Ensure we catch on-disk FIFOs and unix domain sockets.
            file.file.metadata().is_ok_and(|metadata| metadata.is_file()).then_some(file)
        };
        if is_dir(&fd_in) || is_dir(&fd_out) {
            return ecx.set_errno_and_return_neg1(LibcError("EISDIR"), dest);
        }
        let (Some(file_in), Some(file_out)) = (as_regular_file(fd_in), as_regular_file(fd_out))
        else {
            return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
        };
        if !file_in.readable || !file_out.writable || file_out.append {
            // The docs explicitly say that O_APPEND on the output file is forbidden.
            return ecx.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        }
        assert!(ecx.machine.communicate(), "isolation should have prevented even opening a file");

        // Non-deterministically decide to further reduce the length, simulating a partial copy (but
        // never to 0, that would indicate EOF).
        let len = if ecx.machine.short_fd_operations
            && (file_in.short_fd_operations() || file_out.short_fd_operations())
            && len >= 2
            && let rng = ecx.machine.rng.get_mut()
            // Big accesses are shortened with a chance of 75%.
            && (rng.random() || (len >= BIG_ACCESS && rng.random()))
        {
            // Also cap length to remaining input length as otherwise halving it does nothing.
            let mut input = &file_in.file;
            let remaining_bytes =
                try { input.stream_len()?.saturating_sub(input.stream_position()?).clamp_to(2..) };
            let len = len.clamp_to(..=remaining_bytes.unwrap_or(u64::MAX));
            len / 2 // since `len` is at least 2, the result is still at least 1
        } else {
            len
        };

        let copy = || -> io::Result<u64> {
            let (mut input, mut output) = (&file_in.file, &file_out.file);
            let (pos_in, pos_out) = (input.stream_position()?, output.stream_position()?);

            // Error if both FDs refer to the same file and the ranges overlap.
            let (meta_in, meta_out) = (input.metadata()?, output.metadata()?);
            if same_host_file(&meta_in, &meta_out)
                && (pos_in.checked_add(len).is_none_or(|end_in| end_in > pos_out)
                    && pos_out.checked_add(len).is_none_or(|end_out| end_out > pos_in))
            {
                return Err(io::ErrorKind::InvalidInput.into());
            }

            io::copy(&mut input.take(len), &mut output).inspect_err(|_| {
                // Like Linux, leave the file positions unchanged on error. See
                // <https://github.com/torvalds/linux/blob/81dfa035891f0502674332ee97db348a16486de0/fs/read_write.c#L1686-L1703>.
                let _ = input.seek(SeekFrom::Start(pos_in));
                let _ = output.seek(SeekFrom::Start(pos_out));
            })
        };
        ecx.return_io_result_via_errno(copy(), dest)
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

fn file_type_to_mode_name(file_type: std::fs::FileType) -> &'static str {
    if file_type.is_file() {
        "S_IFREG"
    } else if file_type.is_dir() {
        "S_IFDIR"
    } else if file_type.is_symlink() {
        "S_IFLNK"
    } else {
        // Certain file types are only available when the host is a Unix system.
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileTypeExt;
            if file_type.is_socket() {
                return "S_IFSOCK";
            } else if file_type.is_fifo() {
                return "S_IFIFO";
            } else if file_type.is_char_device() {
                return "S_IFCHR";
            } else if file_type.is_block_device() {
                return "S_IFBLK";
            }
        }
        "S_IFREG"
    }
}

/// Stores a file's metadata in order to avoid code duplication in the different metadata related
/// shims.
///
/// Some fields are host/platform-specific. `None` means that Miri does not have a real value for
/// this field, for example because the metadata is synthetic or because the host platform does not
/// expose it. `statx` must only advertise the corresponding `STATX_*` bit when the field is `Some`;
/// legacy `stat` writes zero for `None` to preserve the old fallback behavior.
struct FileMetadata {
    /// This holds both the file type (dir, regular, symlink, ...) and permissions.
    mode: u32,
    size: u64,
    created: Option<(u64, u32)>,
    accessed: Option<(u64, u32)>,
    modified: Option<(u64, u32)>,
    dev: Option<u64>,
    ino: Option<u64>,
    nlink: Option<u64>,
    uid: Option<u32>,
    gid: Option<u32>,
    blksize: Option<u64>,
    blocks: Option<u64>,
}

impl FileMetadata {
    /// Implements the shared "metadata at" semantics of `fstatat` and `statx`.
    fn at<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        dirfd: FdNum,
        path: &Path,
        symlink_nofollow_flag: bool,
        empty_path_flag: bool,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        // If the path is empty, and the AT_EMPTY_PATH flag is set, we query the open file
        // represented by dirfd, whether it's a directory or otherwise.
        if empty_path_flag && path.is_empty() {
            return FileMetadata::from_fd_num(ecx, dirfd);
        }

        let dir = match ecx.get_dir(Some(dirfd), path) {
            Ok(dir) => dir,
            Err(err) => return interp_ok(Err(err)),
        };
        // Windows does not support "." relative to a `Dir` handle, but the empty path works.
        // (https://github.com/rust-lang/rust/issues/163032)
        let path = if cfg!(windows) && dir.is_some() && path.to_str() == Some(".") {
            Path::new("")
        } else {
            path
        };

        let metadata = match dir {
            None =>
                if symlink_nofollow_flag {
                    path.symlink_metadata()
                } else {
                    path.metadata()
                },
            Some(dir) =>
                if symlink_nofollow_flag {
                    dir.dir.symlink_metadata(path)
                } else {
                    dir.dir.metadata(path)
                },
        };

        FileMetadata::from_host(ecx, metadata)
    }

    fn from_fd_num<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        fd_num: FdNum,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        let Some(fd) = ecx.machine.fds.get(fd_num) else {
            return interp_ok(Err(LibcError("EBADF")));
        };
        match fd.metadata(ecx.machine.communicate())? {
            Either::Left(host) => Self::from_host(ecx, host),
            Either::Right(mode_name) => Self::synthetic(ecx, mode_name),
        }
    }

    fn synthetic<'tcx>(
        ecx: &mut MiriInterpCx<'tcx>,
        mode_name: &str,
    ) -> InterpResult<'tcx, Result<FileMetadata, IoError>> {
        let mode = ecx.eval_libc_int::<u32>(mode_name);
        // We observed 0x777 on sockets and 0x600 on pipes...
        let mode = mode | 0o666;
        interp_ok(Ok(FileMetadata {
            mode,
            size: 0,
            created: None,
            accessed: None,
            modified: None,
            dev: None,
            uid: None,
            gid: None,
            blksize: None,
            blocks: None,
            ino: None,
            nlink: None,
        }))
    }

    fn from_host<'tcx>(
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
        let mut mode = ecx.eval_libc_int::<u32>(file_type_to_mode_name(file_type));

        let size = metadata.len();

        let created = extract_sec_and_nsec(metadata.created())?;
        let accessed = extract_sec_and_nsec(metadata.accessed())?;
        let modified = extract_sec_and_nsec(metadata.modified())?;

        // FIXME: Provide more fields using platform specific methods.

        cfg_select! {
            unix => {
                use std::os::unix::fs::{MetadataExt, PermissionsExt};

                let dev = metadata.dev();
                let ino = metadata.ino();
                let nlink = metadata.nlink();
                let uid = metadata.uid();
                let gid = metadata.gid();
                let blksize = metadata.blksize();
                let blocks = metadata.blocks();

                mode |= metadata.permissions().mode();

                interp_ok(Ok(FileMetadata {
                    mode,
                    size,
                    created,
                    accessed,
                    modified,
                    dev: Some(dev),
                    ino: Some(ino),
                    nlink: Some(nlink),
                    uid: Some(uid),
                    gid: Some(gid),
                    blksize: Some(blksize),
                    blocks: Some(blocks),
                }))
            }
            _ => {
                // Emulate "everyone can read" or "everyone can read and write".
                mode |= if metadata.permissions().readonly() { 0o444 } else { 0o666 };

                interp_ok(Ok(FileMetadata {
                    mode,
                    size,
                    created,
                    accessed,
                    modified,
                    dev: None,
                    ino: None,
                    nlink: None,
                    uid: None,
                    gid: None,
                    blksize: None,
                    blocks: None,
                }))
            }
        }
    }
}
