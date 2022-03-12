use std::borrow::Cow;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::fs::{
    read_dir, remove_dir, remove_file, rename, DirBuilder, File, FileType, OpenOptions, ReadDir,
};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::SystemTime;

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{self, layout::LayoutOf};
use rustc_target::abi::{Align, Size};

use crate::*;
use helpers::check_arg_count;
use shims::os_str::os_str_to_bytes;
use shims::time::system_time_to_duration;

#[derive(Debug)]
struct FileHandle {
    file: File,
    writable: bool,
}

trait FileDescriptor: std::fmt::Debug {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle>;

    fn read<'tcx>(
        &mut self,
        communicate_allowed: bool,
        bytes: &mut [u8],
    ) -> InterpResult<'tcx, io::Result<usize>>;
    fn write<'tcx>(
        &mut self,
        communicate_allowed: bool,
        bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>>;
    fn seek<'tcx>(
        &mut self,
        communicate_allowed: bool,
        offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>>;
    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>>;

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>>;
}

impl FileDescriptor for FileHandle {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle> {
        Ok(&self)
    }

    fn read<'tcx>(
        &mut self,
        communicate_allowed: bool,
        bytes: &mut [u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.read(bytes))
    }

    fn write<'tcx>(
        &mut self,
        communicate_allowed: bool,
        bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.write(bytes))
    }

    fn seek<'tcx>(
        &mut self,
        communicate_allowed: bool,
        offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.seek(offset))
    }

    fn close<'tcx>(
        self: Box<Self>,
        communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        // We sync the file if it was opened in a mode different than read-only.
        if self.writable {
            // `File::sync_all` does the checks that are done when closing a file. We do this to
            // to handle possible errors correctly.
            let result = self.file.sync_all().map(|_| 0i32);
            // Now we actually close the file.
            drop(self);
            // And return the result.
            Ok(result)
        } else {
            // We drop the file, this closes it but ignores any errors
            // produced when closing it. This is done because
            // `File::sync_all` cannot be done over files like
            // `/dev/urandom` which are read-only. Check
            // https://github.com/rust-lang/miri/issues/999#issuecomment-568920439
            // for a deeper discussion.
            drop(self);
            Ok(Ok(0))
        }
    }

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        let duplicated = self.file.try_clone()?;
        Ok(Box::new(FileHandle { file: duplicated, writable: self.writable }))
    }
}

impl FileDescriptor for io::Stdin {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle> {
        throw_unsup_format!("stdin cannot be used as FileHandle");
    }

    fn read<'tcx>(
        &mut self,
        communicate_allowed: bool,
        bytes: &mut [u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        if !communicate_allowed {
            // We want isolation mode to be deterministic, so we have to disallow all reads, even stdin.
            helpers::isolation_abort_error("`read` from stdin")?;
        }
        Ok(Read::read(self, bytes))
    }

    fn write<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot write to stdin");
    }

    fn seek<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stdin");
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        throw_unsup_format!("stdin cannot be closed");
    }

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(io::stdin()))
    }
}

impl FileDescriptor for io::Stdout {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle> {
        throw_unsup_format!("stdout cannot be used as FileHandle");
    }

    fn read<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _bytes: &mut [u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot read from stdout");
    }

    fn write<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        // We allow writing to stderr even with isolation enabled.
        let result = Write::write(self, bytes);
        // Stdout is buffered, flush to make sure it appears on the
        // screen.  This is the write() syscall of the interpreted
        // program, we want it to correspond to a write() syscall on
        // the host -- there is no good in adding extra buffering
        // here.
        io::stdout().flush().unwrap();

        Ok(result)
    }

    fn seek<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stdout");
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        throw_unsup_format!("stdout cannot be closed");
    }

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(io::stdout()))
    }
}

impl FileDescriptor for io::Stderr {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle> {
        throw_unsup_format!("stderr cannot be used as FileHandle");
    }

    fn read<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _bytes: &mut [u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot read from stderr");
    }

    fn write<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        bytes: &[u8],
    ) -> InterpResult<'tcx, io::Result<usize>> {
        // We allow writing to stderr even with isolation enabled.
        // No need to flush, stderr is not buffered.
        Ok(Write::write(self, bytes))
    }

    fn seek<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        _offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stderr");
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<i32>> {
        throw_unsup_format!("stderr cannot be closed");
    }

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>> {
        Ok(Box::new(io::stderr()))
    }
}

#[derive(Debug)]
pub struct FileHandler {
    handles: BTreeMap<i32, Box<dyn FileDescriptor>>,
}

impl<'tcx> Default for FileHandler {
    fn default() -> Self {
        let mut handles: BTreeMap<_, Box<dyn FileDescriptor>> = BTreeMap::new();
        handles.insert(0i32, Box::new(io::stdin()));
        handles.insert(1i32, Box::new(io::stdout()));
        handles.insert(2i32, Box::new(io::stderr()));
        FileHandler { handles }
    }
}

impl<'tcx> FileHandler {
    fn insert_fd(&mut self, file_handle: Box<dyn FileDescriptor>) -> i32 {
        self.insert_fd_with_min_fd(file_handle, 0)
    }

    fn insert_fd_with_min_fd(&mut self, file_handle: Box<dyn FileDescriptor>, min_fd: i32) -> i32 {
        // Find the lowest unused FD, starting from min_fd. If the first such unused FD is in
        // between used FDs, the find_map combinator will return it. If the first such unused FD
        // is after all other used FDs, the find_map combinator will return None, and we will use
        // the FD following the greatest FD thus far.
        let candidate_new_fd =
            self.handles.range(min_fd..).zip(min_fd..).find_map(|((fd, _fh), counter)| {
                if *fd != counter {
                    // There was a gap in the fds stored, return the first unused one
                    // (note that this relies on BTreeMap iterating in key order)
                    Some(counter)
                } else {
                    // This fd is used, keep going
                    None
                }
            });
        let new_fd = candidate_new_fd.unwrap_or_else(|| {
            // find_map ran out of BTreeMap entries before finding a free fd, use one plus the
            // maximum fd in the map
            self.handles
                .last_key_value()
                .map(|(fd, _)| fd.checked_add(1).unwrap())
                .unwrap_or(min_fd)
        });

        self.handles.try_insert(new_fd, file_handle).unwrap();
        new_fd
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExtPrivate<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
trait EvalContextExtPrivate<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn macos_stat_write_buf(
        &mut self,
        metadata: FileMetadata,
        buf_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mode: u16 = metadata.mode.to_u16()?;

        let (access_sec, access_nsec) = metadata.accessed.unwrap_or((0, 0));
        let (created_sec, created_nsec) = metadata.created.unwrap_or((0, 0));
        let (modified_sec, modified_nsec) = metadata.modified.unwrap_or((0, 0));

        let buf = this.deref_operand(buf_op)?;
        this.write_int_fields_named(
            &[
                ("st_dev", 0),
                ("st_mode", mode.into()),
                ("st_nlink", 0),
                ("st_ino", 0),
                ("st_uid", 0),
                ("st_gid", 0),
                ("st_rdev", 0),
                ("st_atime", access_sec.into()),
                ("st_atime_nsec", access_nsec.into()),
                ("st_mtime", modified_sec.into()),
                ("st_mtime_nsec", modified_nsec.into()),
                ("st_ctime", 0),
                ("st_ctime_nsec", 0),
                ("st_birthtime", created_sec.into()),
                ("st_birthtime_nsec", created_nsec.into()),
                ("st_size", metadata.size.into()),
                ("st_blocks", 0),
                ("st_blksize", 0),
                ("st_flags", 0),
                ("st_gen", 0),
            ],
            &buf,
        )?;

        Ok(0)
    }

    /// Function used when a handle is not found inside `FileHandler`. It returns `Ok(-1)`and sets
    /// the last OS error to `libc::EBADF` (invalid file descriptor). This function uses
    /// `T: From<i32>` instead of `i32` directly because some fs functions return different integer
    /// types (like `read`, that returns an `i64`).
    fn handle_not_found<T: From<i32>>(&mut self) -> InterpResult<'tcx, T> {
        let this = self.eval_context_mut();
        let ebadf = this.eval_libc("EBADF")?;
        this.set_last_error(ebadf)?;
        Ok((-1).into())
    }

    fn file_type_to_d_type(
        &mut self,
        file_type: std::io::Result<FileType>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        match file_type {
            Ok(file_type) => {
                if file_type.is_dir() {
                    Ok(this.eval_libc("DT_DIR")?.to_u8()?.into())
                } else if file_type.is_file() {
                    Ok(this.eval_libc("DT_REG")?.to_u8()?.into())
                } else if file_type.is_symlink() {
                    Ok(this.eval_libc("DT_LNK")?.to_u8()?.into())
                } else {
                    // Certain file types are only supported when the host is a Unix system.
                    // (i.e. devices and sockets) If it is, check those cases, if not, fall back to
                    // DT_UNKNOWN sooner.

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::FileTypeExt;
                        if file_type.is_block_device() {
                            Ok(this.eval_libc("DT_BLK")?.to_u8()?.into())
                        } else if file_type.is_char_device() {
                            Ok(this.eval_libc("DT_CHR")?.to_u8()?.into())
                        } else if file_type.is_fifo() {
                            Ok(this.eval_libc("DT_FIFO")?.to_u8()?.into())
                        } else if file_type.is_socket() {
                            Ok(this.eval_libc("DT_SOCK")?.to_u8()?.into())
                        } else {
                            Ok(this.eval_libc("DT_UNKNOWN")?.to_u8()?.into())
                        }
                    }
                    #[cfg(not(unix))]
                    Ok(this.eval_libc("DT_UNKNOWN")?.to_u8()?.into())
                }
            }
            Err(e) =>
                return match e.raw_os_error() {
                    Some(error) => Ok(error),
                    None =>
                        throw_unsup_format!(
                            "the error {} couldn't be converted to a return value",
                            e
                        ),
                },
        }
    }
}

/// An open directory, tracked by DirHandler.
#[derive(Debug)]
pub struct OpenDir {
    /// The directory reader on the host.
    read_dir: ReadDir,
    /// The most recent entry returned by readdir()
    entry: Pointer<Option<Tag>>,
}

impl OpenDir {
    fn new(read_dir: ReadDir) -> Self {
        // We rely on `free` being a NOP on null pointers.
        Self { read_dir, entry: Pointer::null() }
    }
}

#[derive(Debug)]
pub struct DirHandler {
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

impl DirHandler {
    fn insert_new(&mut self, read_dir: ReadDir) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.streams.try_insert(id, OpenDir::new(read_dir)).unwrap();
        id
    }
}

impl Default for DirHandler {
    fn default() -> DirHandler {
        DirHandler {
            streams: FxHashMap::default(),
            // Skip 0 as an ID, because it looks like a null pointer to libc
            next_id: 1,
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

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn open(&mut self, args: &[OpTy<'tcx, Tag>]) -> InterpResult<'tcx, i32> {
        if args.len() < 2 || args.len() > 3 {
            throw_ub_format!(
                "incorrect number of arguments for `open`: got {}, expected 2 or 3",
                args.len()
            );
        }

        let this = self.eval_context_mut();

        let path_op = &args[0];
        let flag = this.read_scalar(&args[1])?.to_i32()?;

        let mut options = OpenOptions::new();

        let o_rdonly = this.eval_libc_i32("O_RDONLY")?;
        let o_wronly = this.eval_libc_i32("O_WRONLY")?;
        let o_rdwr = this.eval_libc_i32("O_RDWR")?;
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

        let o_append = this.eval_libc_i32("O_APPEND")?;
        if flag & o_append != 0 {
            options.append(true);
            mirror |= o_append;
        }
        let o_trunc = this.eval_libc_i32("O_TRUNC")?;
        if flag & o_trunc != 0 {
            options.truncate(true);
            mirror |= o_trunc;
        }
        let o_creat = this.eval_libc_i32("O_CREAT")?;
        if flag & o_creat != 0 {
            // Get the mode.  On macOS, the argument type `mode_t` is actually `u16`, but
            // C integer promotion rules mean that on the ABI level, it gets passed as `u32`
            // (see https://github.com/rust-lang/rust/issues/71915).
            let mode = if let Some(arg) = args.get(2) {
                this.read_scalar(arg)?.to_u32()?
            } else {
                throw_ub_format!(
                    "incorrect number of arguments for `open` with `O_CREAT`: got {}, expected 3",
                    args.len()
                );
            };

            if mode != 0o666 {
                throw_unsup_format!("non-default mode 0o{:o} is not supported", mode);
            }

            mirror |= o_creat;

            let o_excl = this.eval_libc_i32("O_EXCL")?;
            if flag & o_excl != 0 {
                mirror |= o_excl;
                options.create_new(true);
            } else {
                options.create(true);
            }
        }
        let o_cloexec = this.eval_libc_i32("O_CLOEXEC")?;
        if flag & o_cloexec != 0 {
            // We do not need to do anything for this flag because `std` already sets it.
            // (Technically we do not support *not* setting this flag, but we ignore that.)
            mirror |= o_cloexec;
        }
        // If `flag` is not equal to `mirror`, there is an unsupported option enabled in `flag`,
        // then we throw an error.
        if flag != mirror {
            throw_unsup_format!("unsupported flags {:#x}", flag & !mirror);
        }

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`open`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        let fd = options.open(&path).map(|file| {
            let fh = &mut this.machine.file_handler;
            fh.insert_fd(Box::new(FileHandle { file, writable }))
        });

        this.try_unwrap_io_result(fd)
    }

    fn fcntl(&mut self, args: &[OpTy<'tcx, Tag>]) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if args.len() < 2 {
            throw_ub_format!(
                "incorrect number of arguments for fcntl: got {}, expected at least 2",
                args.len()
            );
        }
        let fd = this.read_scalar(&args[0])?.to_i32()?;
        let cmd = this.read_scalar(&args[1])?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fcntl`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        // We only support getting the flags for a descriptor.
        if cmd == this.eval_libc_i32("F_GETFD")? {
            // Currently this is the only flag that `F_GETFD` returns. It is OK to just return the
            // `FD_CLOEXEC` value without checking if the flag is set for the file because `std`
            // always sets this flag when opening a file. However we still need to check that the
            // file itself is open.
            let &[_, _] = check_arg_count(args)?;
            if this.machine.file_handler.handles.contains_key(&fd) {
                Ok(this.eval_libc_i32("FD_CLOEXEC")?)
            } else {
                this.handle_not_found()
            }
        } else if cmd == this.eval_libc_i32("F_DUPFD")?
            || cmd == this.eval_libc_i32("F_DUPFD_CLOEXEC")?
        {
            // Note that we always assume the FD_CLOEXEC flag is set for every open file, in part
            // because exec() isn't supported. The F_DUPFD and F_DUPFD_CLOEXEC commands only
            // differ in whether the FD_CLOEXEC flag is pre-set on the new file descriptor,
            // thus they can share the same implementation here.
            let &[_, _, ref start] = check_arg_count(args)?;
            let start = this.read_scalar(start)?.to_i32()?;

            let fh = &mut this.machine.file_handler;

            match fh.handles.get_mut(&fd) {
                Some(file_descriptor) => {
                    let dup_result = file_descriptor.dup();
                    match dup_result {
                        Ok(dup_fd) => Ok(fh.insert_fd_with_min_fd(dup_fd, start)),
                        Err(e) => {
                            this.set_last_error_from_io_error(e.kind())?;
                            Ok(-1)
                        }
                    }
                }
                None => return this.handle_not_found(),
            }
        } else if this.tcx.sess.target.os == "macos" && cmd == this.eval_libc_i32("F_FULLFSYNC")? {
            let &[_, _] = check_arg_count(args)?;
            if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
                // FIXME: Support fullfsync for all FDs
                let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
                let io_result = maybe_sync_file(&file, *writable, File::sync_all);
                this.try_unwrap_io_result(io_result)
            } else {
                this.handle_not_found()
            }
        } else {
            throw_unsup_format!("the {:#x} command is not supported for `fcntl`)", cmd);
        }
    }

    fn close(&mut self, fd_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        if let Some(file_descriptor) = this.machine.file_handler.handles.remove(&fd) {
            let result = file_descriptor.close(this.machine.communicate())?;
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn read(&mut self, fd: i32, buf: Pointer<Option<Tag>>, count: u64) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescriptor` trait.

        trace!("Reading from FD {}, size {}", fd, count);

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access_align(
            buf,
            Size::from_bytes(count),
            Align::ONE,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.machine_isize_max() as u64).min(isize::MAX as u64);
        let communicate = this.machine.communicate();

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            trace!("read: FD mapped to {:?}", file_descriptor);
            // We want to read at most `count` bytes. We are sure that `count` is not negative
            // because it was a target's `usize`. Also we are sure that its smaller than
            // `usize::MAX` because it is a host's `isize`.
            let mut bytes = vec![0; count as usize];
            // `File::read` never returns a value larger than `count`,
            // so this cannot fail.
            let result =
                file_descriptor.read(communicate, &mut bytes)?.map(|c| i64::try_from(c).unwrap());

            match result {
                Ok(read_bytes) => {
                    // If reading to `bytes` did not fail, we write those bytes to the buffer.
                    this.memory.write_bytes(buf, bytes)?;
                    Ok(read_bytes)
                }
                Err(e) => {
                    this.set_last_error_from_io_error(e.kind())?;
                    Ok(-1)
                }
            }
        } else {
            trace!("read: FD not found");
            this.handle_not_found()
        }
    }

    fn write(&mut self, fd: i32, buf: Pointer<Option<Tag>>, count: u64) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescriptor` trait.

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access_align(
            buf,
            Size::from_bytes(count),
            Align::ONE,
            CheckInAllocMsg::MemoryAccessTest,
        )?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.machine_isize_max() as u64).min(isize::MAX as u64);
        let communicate = this.machine.communicate();

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            let bytes = this.memory.read_bytes(buf, Size::from_bytes(count))?;
            let result =
                file_descriptor.write(communicate, &bytes)?.map(|c| i64::try_from(c).unwrap());
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn lseek64(
        &mut self,
        fd_op: &OpTy<'tcx, Tag>,
        offset_op: &OpTy<'tcx, Tag>,
        whence_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescriptor` trait.

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let offset = this.read_scalar(offset_op)?.to_i64()?;
        let whence = this.read_scalar(whence_op)?.to_i32()?;

        let seek_from = if whence == this.eval_libc_i32("SEEK_SET")? {
            SeekFrom::Start(u64::try_from(offset).unwrap())
        } else if whence == this.eval_libc_i32("SEEK_CUR")? {
            SeekFrom::Current(offset)
        } else if whence == this.eval_libc_i32("SEEK_END")? {
            SeekFrom::End(offset)
        } else {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        };

        let communicate = this.machine.communicate();
        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            let result = file_descriptor
                .seek(communicate, seek_from)?
                .map(|offset| i64::try_from(offset).unwrap());
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn unlink(&mut self, path_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`unlink`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        let result = remove_file(path).map(|_| 0);
        this.try_unwrap_io_result(result)
    }

    fn symlink(
        &mut self,
        target_op: &OpTy<'tcx, Tag>,
        linkpath_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
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
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        let result = create_link(&target, &linkpath).map(|_| 0);
        this.try_unwrap_io_result(result)
    }

    fn macos_stat(
        &mut self,
        path_op: &OpTy<'tcx, Tag>,
        buf_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("macos", "stat");

        let path_scalar = this.read_pointer(path_op)?;
        let path = this.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`stat`", reject_with)?;
            let eacc = this.eval_libc("EACCES")?;
            this.set_last_error(eacc)?;
            return Ok(-1);
        }

        // `stat` always follows symlinks.
        let metadata = match FileMetadata::from_path(this, &path, true)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };

        this.macos_stat_write_buf(metadata, buf_op)
    }

    // `lstat` is used to get symlink metadata.
    fn macos_lstat(
        &mut self,
        path_op: &OpTy<'tcx, Tag>,
        buf_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("macos", "lstat");

        let path_scalar = this.read_pointer(path_op)?;
        let path = this.read_path_from_c_str(path_scalar)?.into_owned();

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`lstat`", reject_with)?;
            let eacc = this.eval_libc("EACCES")?;
            this.set_last_error(eacc)?;
            return Ok(-1);
        }

        let metadata = match FileMetadata::from_path(this, &path, false)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };

        this.macos_stat_write_buf(metadata, buf_op)
    }

    fn macos_fstat(
        &mut self,
        fd_op: &OpTy<'tcx, Tag>,
        buf_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "fstat");

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fstat`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        let metadata = match FileMetadata::from_fd(this, fd)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };
        this.macos_stat_write_buf(metadata, buf_op)
    }

    fn linux_statx(
        &mut self,
        dirfd_op: &OpTy<'tcx, Tag>,    // Should be an `int`
        pathname_op: &OpTy<'tcx, Tag>, // Should be a `const char *`
        flags_op: &OpTy<'tcx, Tag>,    // Should be an `int`
        _mask_op: &OpTy<'tcx, Tag>,    // Should be an `unsigned int`
        statxbuf_op: &OpTy<'tcx, Tag>, // Should be a `struct statx *`
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "statx");

        let statxbuf_ptr = this.read_pointer(statxbuf_op)?;
        let pathname_ptr = this.read_pointer(pathname_op)?;

        // If the statxbuf or pathname pointers are null, the function fails with `EFAULT`.
        if this.ptr_is_null(statxbuf_ptr)? || this.ptr_is_null(pathname_ptr)? {
            let efault = this.eval_libc("EFAULT")?;
            this.set_last_error(efault)?;
            return Ok(-1);
        }

        // Under normal circumstances, we would use `deref_operand(statxbuf_op)` to produce a
        // proper `MemPlace` and then write the results of this function to it. However, the
        // `syscall` function is untyped. This means that all the `statx` parameters are provided
        // as `isize`s instead of having the proper types. Thus, we have to recover the layout of
        // `statxbuf_op` by using the `libc::statx` struct type.
        let statxbuf = {
            // FIXME: This long path is required because `libc::statx` is an struct and also a
            // function and `resolve_path` is returning the latter.
            let statx_ty = this
                .resolve_path(&["libc", "unix", "linux_like", "linux", "gnu", "statx"])
                .ty(*this.tcx, ty::ParamEnv::reveal_all());
            let statx_layout = this.layout_of(statx_ty)?;
            MPlaceTy::from_aligned_ptr(statxbuf_ptr, statx_layout)
        };

        let path = this.read_path_from_c_str(pathname_ptr)?.into_owned();
        // See <https://github.com/rust-lang/rust/pull/79196> for a discussion of argument sizes.
        let flags = this.read_scalar(flags_op)?.to_i32()?;
        let empty_path_flag = flags & this.eval_libc("AT_EMPTY_PATH")?.to_i32()? != 0;
        let dirfd = this.read_scalar(dirfd_op)?.to_i32()?;
        // We only support:
        // * interpreting `path` as an absolute directory,
        // * interpreting `path` as a path relative to `dirfd` when the latter is `AT_FDCWD`, or
        // * interpreting `dirfd` as any file descriptor when `path` is empty and AT_EMPTY_PATH is
        // set.
        // Other behaviors cannot be tested from `libstd` and thus are not implemented. If you
        // found this error, please open an issue reporting it.
        if !(path.is_absolute()
            || dirfd == this.eval_libc_i32("AT_FDCWD")?
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
            let ecode = if path.is_absolute() || dirfd == this.eval_libc_i32("AT_FDCWD")? {
                // since `path` is provided, either absolute or
                // relative to CWD, `EACCES` is the most relevant.
                this.eval_libc("EACCES")?
            } else {
                // `dirfd` is set to target file, and `path` is empty
                // (or we would have hit the `throw_unsup_format`
                // above). `EACCES` would violate the spec.
                assert!(empty_path_flag);
                this.eval_libc("EBADF")?
            };
            this.set_last_error(ecode)?;
            return Ok(-1);
        }

        // the `_mask_op` paramter specifies the file information that the caller requested.
        // However `statx` is allowed to return information that was not requested or to not
        // return information that was requested. This `mask` represents the information we can
        // actually provide for any target.
        let mut mask =
            this.eval_libc("STATX_TYPE")?.to_u32()? | this.eval_libc("STATX_SIZE")?.to_u32()?;

        // If the `AT_SYMLINK_NOFOLLOW` flag is set, we query the file's metadata without following
        // symbolic links.
        let follow_symlink = flags & this.eval_libc("AT_SYMLINK_NOFOLLOW")?.to_i32()? == 0;

        // If the path is empty, and the AT_EMPTY_PATH flag is set, we query the open file
        // represented by dirfd, whether it's a directory or otherwise.
        let metadata = if path.as_os_str().is_empty() && empty_path_flag {
            FileMetadata::from_fd(this, dirfd)?
        } else {
            FileMetadata::from_path(this, &path, follow_symlink)?
        };
        let metadata = match metadata {
            Some(metadata) => metadata,
            None => return Ok(-1),
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
                mask |= this.eval_libc("STATX_ATIME")?.to_u32()?;
                InterpResult::Ok(tup)
            })
            .unwrap_or(Ok((0, 0)))?;

        let (created_sec, created_nsec) = metadata
            .created
            .map(|tup| {
                mask |= this.eval_libc("STATX_BTIME")?.to_u32()?;
                InterpResult::Ok(tup)
            })
            .unwrap_or(Ok((0, 0)))?;

        let (modified_sec, modified_nsec) = metadata
            .modified
            .map(|tup| {
                mask |= this.eval_libc("STATX_MTIME")?.to_u32()?;
                InterpResult::Ok(tup)
            })
            .unwrap_or(Ok((0, 0)))?;

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
        this.write_int_fields(
            &[
                access_sec.into(),  // stx_atime.tv_sec
                access_nsec.into(), // stx_atime.tv_nsec
            ],
            &this.mplace_field_named(&statxbuf, "stx_atime")?,
        )?;
        this.write_int_fields(
            &[
                created_sec.into(),  // stx_btime.tv_sec
                created_nsec.into(), // stx_btime.tv_nsec
            ],
            &this.mplace_field_named(&statxbuf, "stx_btime")?,
        )?;
        this.write_int_fields(
            &[
                0.into(), // stx_ctime.tv_sec
                0.into(), // stx_ctime.tv_nsec
            ],
            &this.mplace_field_named(&statxbuf, "stx_ctime")?,
        )?;
        this.write_int_fields(
            &[
                modified_sec.into(),  // stx_mtime.tv_sec
                modified_nsec.into(), // stx_mtime.tv_nsec
            ],
            &this.mplace_field_named(&statxbuf, "stx_mtime")?,
        )?;

        Ok(0)
    }

    fn rename(
        &mut self,
        oldpath_op: &OpTy<'tcx, Tag>,
        newpath_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let oldpath_ptr = this.read_pointer(oldpath_op)?;
        let newpath_ptr = this.read_pointer(newpath_op)?;

        if this.ptr_is_null(oldpath_ptr)? || this.ptr_is_null(newpath_ptr)? {
            let efault = this.eval_libc("EFAULT")?;
            this.set_last_error(efault)?;
            return Ok(-1);
        }

        let oldpath = this.read_path_from_c_str(oldpath_ptr)?;
        let newpath = this.read_path_from_c_str(newpath_ptr)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`rename`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        let result = rename(oldpath, newpath).map(|_| 0);

        this.try_unwrap_io_result(result)
    }

    fn mkdir(
        &mut self,
        path_op: &OpTy<'tcx, Tag>,
        mode_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        #[cfg_attr(not(unix), allow(unused_variables))]
        let mode = if this.tcx.sess.target.os == "macos" {
            u32::from(this.read_scalar(mode_op)?.to_u16()?)
        } else {
            this.read_scalar(mode_op)?.to_u32()?
        };

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`mkdir`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = DirBuilder::new();

        // If the host supports it, forward on the mode of the directory
        // (i.e. permission bits and the sticky bit)
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(mode.into());
        }

        let result = builder.create(path).map(|_| 0i32);

        this.try_unwrap_io_result(result)
    }

    fn rmdir(&mut self, path_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`rmdir`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(-1);
        }

        let result = remove_dir(path).map(|_| 0i32);

        this.try_unwrap_io_result(result)
    }

    fn opendir(&mut self, name_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        let name = this.read_path_from_c_str(this.read_pointer(name_op)?)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`opendir`", reject_with)?;
            let eacc = this.eval_libc("EACCES")?;
            this.set_last_error(eacc)?;
            return Ok(Scalar::null_ptr(this));
        }

        let result = read_dir(name);

        match result {
            Ok(dir_iter) => {
                let id = this.machine.dir_handler.insert_new(dir_iter);

                // The libc API for opendir says that this method returns a pointer to an opaque
                // structure, but we are returning an ID number. Thus, pass it as a scalar of
                // pointer width.
                Ok(Scalar::from_machine_usize(id, this))
            }
            Err(e) => {
                this.set_last_error_from_io_error(e.kind())?;
                Ok(Scalar::null_ptr(this))
            }
        }
    }

    fn linux_readdir64(&mut self, dirp_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "readdir64");

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readdir`", reject_with)?;
            let eacc = this.eval_libc("EBADF")?;
            this.set_last_error(eacc)?;
            return Ok(Scalar::null_ptr(this));
        }

        let open_dir = this.machine.dir_handler.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir64 did not come from opendir")
        })?;

        let entry = match open_dir.read_dir.next() {
            Some(Ok(dir_entry)) => {
                // Write the directory entry into a newly allocated buffer.
                // The name is written with write_bytes, while the rest of the
                // dirent64 struct is written using write_int_fields.

                // For reference:
                // pub struct dirent64 {
                //     pub d_ino: ino64_t,
                //     pub d_off: off64_t,
                //     pub d_reclen: c_ushort,
                //     pub d_type: c_uchar,
                //     pub d_name: [c_char; 256],
                // }

                let mut name = dir_entry.file_name(); // not a Path as there are no separators!
                name.push("\0"); // Add a NUL terminator
                let name_bytes = os_str_to_bytes(&name)?;
                let name_len = u64::try_from(name_bytes.len()).unwrap();

                let dirent64_layout = this.libc_ty_layout("dirent64")?;
                let d_name_offset = dirent64_layout.fields.offset(4 /* d_name */).bytes();
                let size = d_name_offset.checked_add(name_len).unwrap();

                let entry =
                    this.malloc(size, /*zero_init:*/ false, MiriMemoryKind::Runtime)?;

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;

                this.write_int_fields(
                    &[
                        ino.into(),       // d_ino
                        0,                // d_off
                        size.into(),      // d_reclen
                        file_type.into(), // d_type
                    ],
                    &MPlaceTy::from_aligned_ptr(entry, dirent64_layout),
                )?;

                let name_ptr = entry.offset(Size::from_bytes(d_name_offset), this)?;
                this.memory.write_bytes(name_ptr, name_bytes.iter().copied())?;

                entry
            }
            None => {
                // end of stream: return NULL
                Pointer::null()
            }
            Some(Err(e)) => {
                this.set_last_error_from_io_error(e.kind())?;
                Pointer::null()
            }
        };

        let open_dir = this.machine.dir_handler.streams.get_mut(&dirp).unwrap();
        let old_entry = std::mem::replace(&mut open_dir.entry, entry);
        this.free(old_entry, MiriMemoryKind::Runtime)?;

        Ok(Scalar::from_maybe_pointer(entry, this))
    }

    fn macos_readdir_r(
        &mut self,
        dirp_op: &OpTy<'tcx, Tag>,
        entry_op: &OpTy<'tcx, Tag>,
        result_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "readdir_r");

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readdir_r`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        let open_dir = this.machine.dir_handler.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir_r did not come from opendir")
        })?;
        match open_dir.read_dir.next() {
            Some(Ok(dir_entry)) => {
                // Write into entry, write pointer to result, return 0 on success.
                // The name is written with write_os_str_to_c_str, while the rest of the
                // dirent struct is written using write_int_fields.

                // For reference:
                // pub struct dirent {
                //     pub d_ino: u64,
                //     pub d_seekoff: u64,
                //     pub d_reclen: u16,
                //     pub d_namlen: u16,
                //     pub d_type: u8,
                //     pub d_name: [c_char; 1024],
                // }

                let entry_place = this.deref_operand(entry_op)?;
                let name_place = this.mplace_field(&entry_place, 5)?;

                let file_name = dir_entry.file_name(); // not a Path as there are no separators!
                let (name_fits, file_name_len) = this.write_os_str_to_c_str(
                    &file_name,
                    name_place.ptr,
                    name_place.layout.size.bytes(),
                )?;
                if !name_fits {
                    throw_unsup_format!(
                        "a directory entry had a name too large to fit in libc::dirent"
                    );
                }

                let entry_place = this.deref_operand(entry_op)?;

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;

                this.write_int_fields(
                    &[
                        ino.into(),           // d_ino
                        0,                    // d_seekoff
                        0,                    // d_reclen
                        file_name_len.into(), // d_namlen
                        file_type.into(),     // d_type
                    ],
                    &entry_place,
                )?;

                let result_place = this.deref_operand(result_op)?;
                this.write_scalar(this.read_scalar(entry_op)?, &result_place.into())?;

                Ok(0)
            }
            None => {
                // end of stream: return 0, assign *result=NULL
                this.write_null(&this.deref_operand(result_op)?.into())?;
                Ok(0)
            }
            Some(Err(e)) =>
                match e.raw_os_error() {
                    // return positive error number on error
                    Some(error) => Ok(error),
                    None => {
                        throw_unsup_format!(
                            "the error {} couldn't be converted to a return value",
                            e
                        )
                    }
                },
        }
    }

    fn closedir(&mut self, dirp_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`closedir`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        if let Some(open_dir) = this.machine.dir_handler.streams.remove(&dirp) {
            this.free(open_dir.entry, MiriMemoryKind::Runtime)?;
            drop(open_dir);
            Ok(0)
        } else {
            this.handle_not_found()
        }
    }

    fn ftruncate64(
        &mut self,
        fd_op: &OpTy<'tcx, Tag>,
        length_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let length = this.read_scalar(length_op)?.to_i64()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`ftruncate64`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            // FIXME: Support ftruncate64 for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            if *writable {
                if let Ok(length) = length.try_into() {
                    let result = file.set_len(length);
                    this.try_unwrap_io_result(result.map(|_| 0i32))
                } else {
                    let einval = this.eval_libc("EINVAL")?;
                    this.set_last_error(einval)?;
                    Ok(-1)
                }
            } else {
                // The file is not writable
                let einval = this.eval_libc("EINVAL")?;
                this.set_last_error(einval)?;
                Ok(-1)
            }
        } else {
            this.handle_not_found()
        }
    }

    fn fsync(&mut self, fd_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
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
            return this.handle_not_found();
        }

        if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
            // FIXME: Support fsync for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            let io_result = maybe_sync_file(&file, *writable, File::sync_all);
            this.try_unwrap_io_result(io_result)
        } else {
            this.handle_not_found()
        }
    }

    fn fdatasync(&mut self, fd_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`fdatasync`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
            // FIXME: Support fdatasync for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            let io_result = maybe_sync_file(&file, *writable, File::sync_data);
            this.try_unwrap_io_result(io_result)
        } else {
            this.handle_not_found()
        }
    }

    fn sync_file_range(
        &mut self,
        fd_op: &OpTy<'tcx, Tag>,
        offset_op: &OpTy<'tcx, Tag>,
        nbytes_op: &OpTy<'tcx, Tag>,
        flags_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let offset = this.read_scalar(offset_op)?.to_i64()?;
        let nbytes = this.read_scalar(nbytes_op)?.to_i64()?;
        let flags = this.read_scalar(flags_op)?.to_i32()?;

        if offset < 0 || nbytes < 0 {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }
        let allowed_flags = this.eval_libc_i32("SYNC_FILE_RANGE_WAIT_BEFORE")?
            | this.eval_libc_i32("SYNC_FILE_RANGE_WRITE")?
            | this.eval_libc_i32("SYNC_FILE_RANGE_WAIT_AFTER")?;
        if flags & allowed_flags != flags {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`sync_file_range`", reject_with)?;
            // Set error code as "EBADF" (bad fd)
            return this.handle_not_found();
        }

        if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
            // FIXME: Support sync_data_range for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            let io_result = maybe_sync_file(&file, *writable, File::sync_data);
            this.try_unwrap_io_result(io_result)
        } else {
            this.handle_not_found()
        }
    }

    fn readlink(
        &mut self,
        pathname_op: &OpTy<'tcx, Tag>,
        buf_op: &OpTy<'tcx, Tag>,
        bufsize_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        let pathname = this.read_path_from_c_str(this.read_pointer(pathname_op)?)?;
        let buf = this.read_pointer(buf_op)?;
        let bufsize = this.read_scalar(bufsize_op)?.to_machine_usize(this)?;

        // Reject if isolation is enabled.
        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`readlink`", reject_with)?;
            let eacc = this.eval_libc("EACCES")?;
            this.set_last_error(eacc)?;
            return Ok(-1);
        }

        let result = std::fs::read_link(pathname);
        match result {
            Ok(resolved) => {
                let resolved = this.convert_path_separator(
                    Cow::Borrowed(resolved.as_ref()),
                    crate::shims::os_str::PathConversion::HostToTarget,
                );
                let mut path_bytes = crate::shims::os_str::os_str_to_bytes(resolved.as_ref())?;
                let bufsize: usize = bufsize.try_into().unwrap();
                if path_bytes.len() > bufsize {
                    path_bytes = &path_bytes[..bufsize]
                }
                // 'readlink' truncates the resolved path if
                // the provided buffer is not large enough.
                this.memory.write_bytes(buf, path_bytes.iter().copied())?;
                Ok(path_bytes.len().try_into().unwrap())
            }
            Err(e) => {
                this.set_last_error_from_io_error(e.kind())?;
                Ok(-1)
            }
        }
    }
}

/// Extracts the number of seconds and nanoseconds elapsed between `time` and the unix epoch when
/// `time` is Ok. Returns `None` if `time` is an error. Fails if `time` happens before the unix
/// epoch.
fn extract_sec_and_nsec<'tcx>(
    time: std::io::Result<SystemTime>,
) -> InterpResult<'tcx, Option<(u64, u32)>> {
    time.ok()
        .map(|time| {
            let duration = system_time_to_duration(&time)?;
            Ok((duration.as_secs(), duration.subsec_nanos()))
        })
        .transpose()
}

/// Stores a file's metadata in order to avoid code duplication in the different metadata related
/// shims.
struct FileMetadata {
    mode: Scalar<Tag>,
    size: u64,
    created: Option<(u64, u32)>,
    accessed: Option<(u64, u32)>,
    modified: Option<(u64, u32)>,
}

impl FileMetadata {
    fn from_path<'tcx, 'mir>(
        ecx: &mut MiriEvalContext<'mir, 'tcx>,
        path: &Path,
        follow_symlink: bool,
    ) -> InterpResult<'tcx, Option<FileMetadata>> {
        let metadata =
            if follow_symlink { std::fs::metadata(path) } else { std::fs::symlink_metadata(path) };

        FileMetadata::from_meta(ecx, metadata)
    }

    fn from_fd<'tcx, 'mir>(
        ecx: &mut MiriEvalContext<'mir, 'tcx>,
        fd: i32,
    ) -> InterpResult<'tcx, Option<FileMetadata>> {
        let option = ecx.machine.file_handler.handles.get(&fd);
        let file = match option {
            Some(file_descriptor) => &file_descriptor.as_file_handle()?.file,
            None => return ecx.handle_not_found().map(|_: i32| None),
        };
        let metadata = file.metadata();

        FileMetadata::from_meta(ecx, metadata)
    }

    fn from_meta<'tcx, 'mir>(
        ecx: &mut MiriEvalContext<'mir, 'tcx>,
        metadata: Result<std::fs::Metadata, std::io::Error>,
    ) -> InterpResult<'tcx, Option<FileMetadata>> {
        let metadata = match metadata {
            Ok(metadata) => metadata,
            Err(e) => {
                ecx.set_last_error_from_io_error(e.kind())?;
                return Ok(None);
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

        let mode = ecx.eval_libc(mode_name)?;

        let size = metadata.len();

        let created = extract_sec_and_nsec(metadata.created())?;
        let accessed = extract_sec_and_nsec(metadata.accessed())?;
        let modified = extract_sec_and_nsec(metadata.modified())?;

        // FIXME: Provide more fields using platform specific methods.
        Ok(Some(FileMetadata { mode, size, created, accessed, modified }))
    }
}
