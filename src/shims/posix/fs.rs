use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::fs::{read_dir, remove_dir, remove_file, rename, DirBuilder, File, FileType, OpenOptions, ReadDir};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::SystemTime;

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_target::abi::{Align, LayoutOf, Size};
use rustc_middle::ty;

use crate::*;
use stacked_borrows::Tag;
use helpers::{check_arg_count, immty_from_int_checked, immty_from_uint_checked};
use shims::time::system_time_to_duration;

#[derive(Debug)]
struct FileHandle {
    file: File,
    writable: bool,
}

trait FileDescriptor : std::fmt::Debug {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle>;

    fn read<'tcx>(&mut self, communicate_allowed: bool, bytes: &mut [u8]) -> InterpResult<'tcx, io::Result<usize>>;
    fn write<'tcx>(&mut self, communicate_allowed: bool, bytes: &[u8]) -> InterpResult<'tcx, io::Result<usize>>;
    fn seek<'tcx>(&mut self, communicate_allowed: bool, offset: SeekFrom) -> InterpResult<'tcx, io::Result<u64>>;
    fn close<'tcx>(self: Box<Self>, _communicate_allowed: bool) -> InterpResult<'tcx, io::Result<i32>>;

    fn dup<'tcx>(&mut self) -> io::Result<Box<dyn FileDescriptor>>;
}

impl FileDescriptor for FileHandle {
    fn as_file_handle<'tcx>(&self) -> InterpResult<'tcx, &FileHandle> {
        Ok(&self)
    }

    fn read<'tcx>(&mut self, communicate_allowed: bool, bytes: &mut [u8]) -> InterpResult<'tcx, io::Result<usize>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.read(bytes))
    }

    fn write<'tcx>(&mut self, communicate_allowed: bool, bytes: &[u8]) -> InterpResult<'tcx, io::Result<usize>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.write(bytes))
    }

    fn seek<'tcx>(&mut self, communicate_allowed: bool, offset: SeekFrom) -> InterpResult<'tcx, io::Result<u64>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        Ok(self.file.seek(offset))
    }

    fn close<'tcx>(self: Box<Self>, communicate_allowed: bool) -> InterpResult<'tcx, io::Result<i32>> {
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

    fn read<'tcx>(&mut self, communicate_allowed: bool, bytes: &mut [u8]) -> InterpResult<'tcx, io::Result<usize>> {
        if !communicate_allowed {
            // We want isolation mode to be deterministic, so we have to disallow all reads, even stdin.
            helpers::isolation_error("read")?;
        }
        Ok(Read::read(self, bytes))
    }

    fn write<'tcx>(&mut self, _communicate_allowed: bool, _bytes: &[u8]) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot write to stdin");
    }

    fn seek<'tcx>(&mut self, _communicate_allowed: bool, _offset: SeekFrom) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stdin");
    }

    fn close<'tcx>(self: Box<Self>, _communicate_allowed: bool) -> InterpResult<'tcx, io::Result<i32>> {
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

    fn read<'tcx>(&mut self, _communicate_allowed: bool, _bytes: &mut [u8]) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot read from stdout");
    }

    fn write<'tcx>(&mut self, _communicate_allowed: bool, bytes: &[u8]) -> InterpResult<'tcx, io::Result<usize>> {
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

    fn seek<'tcx>(&mut self, _communicate_allowed: bool, _offset: SeekFrom) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stdout");
    }

    fn close<'tcx>(self: Box<Self>, _communicate_allowed: bool) -> InterpResult<'tcx, io::Result<i32>> {
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

    fn read<'tcx>(&mut self, _communicate_allowed: bool, _bytes: &mut [u8]) -> InterpResult<'tcx, io::Result<usize>> {
        throw_unsup_format!("cannot read from stderr");
    }

    fn write<'tcx>(&mut self, _communicate_allowed: bool, bytes: &[u8]) -> InterpResult<'tcx, io::Result<usize>> {
        // We allow writing to stderr even with isolation enabled.
        // No need to flush, stderr is not buffered.
        Ok(Write::write(self, bytes))
    }

    fn seek<'tcx>(&mut self, _communicate_allowed: bool, _offset: SeekFrom) -> InterpResult<'tcx, io::Result<u64>> {
        throw_unsup_format!("cannot seek on stderr");
    }

    fn close<'tcx>(self: Box<Self>, _communicate_allowed: bool) -> InterpResult<'tcx, io::Result<i32>> {
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
        FileHandler {
            handles
        }
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
        let candidate_new_fd = self
            .handles
            .range(min_fd..)
            .zip(min_fd..)
            .find_map(|((fd, _fh), counter)| {
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
            self.handles.last_key_value().map(|(fd, _)| fd.checked_add(1).unwrap()).unwrap_or(min_fd)
        });

        self.handles.insert(new_fd, file_handle).unwrap_none();
        new_fd
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExtPrivate<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
trait EvalContextExtPrivate<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    /// Emulate `stat` or `lstat` on `macos`. This function is not intended to be
    /// called directly from `emulate_foreign_item_by_name`, so it does not check if isolation is
    /// disabled or if the target OS is the correct one. Please use `macos_stat` or
    /// `macos_lstat` instead.
    fn macos_stat_or_lstat(
        &mut self,
        follow_symlink: bool,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let path_scalar = this.read_scalar(path_op)?.check_init()?;
        let path = this.read_path_from_c_str(path_scalar)?.into_owned();

        let metadata = match FileMetadata::from_path(this, &path, follow_symlink)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };
        this.macos_stat_write_buf(metadata, buf_op)
    }

    fn macos_stat_write_buf(
        &mut self,
        metadata: FileMetadata,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mode: u16 = metadata.mode.to_u16()?;

        let (access_sec, access_nsec) = metadata.accessed.unwrap_or((0, 0));
        let (created_sec, created_nsec) = metadata.created.unwrap_or((0, 0));
        let (modified_sec, modified_nsec) = metadata.modified.unwrap_or((0, 0));

        let dev_t_layout = this.libc_ty_layout("dev_t")?;
        let mode_t_layout = this.libc_ty_layout("mode_t")?;
        let nlink_t_layout = this.libc_ty_layout("nlink_t")?;
        let ino_t_layout = this.libc_ty_layout("ino_t")?;
        let uid_t_layout = this.libc_ty_layout("uid_t")?;
        let gid_t_layout = this.libc_ty_layout("gid_t")?;
        let time_t_layout = this.libc_ty_layout("time_t")?;
        let long_layout = this.libc_ty_layout("c_long")?;
        let off_t_layout = this.libc_ty_layout("off_t")?;
        let blkcnt_t_layout = this.libc_ty_layout("blkcnt_t")?;
        let blksize_t_layout = this.libc_ty_layout("blksize_t")?;
        let uint32_t_layout = this.libc_ty_layout("uint32_t")?;

        let imms = [
            immty_from_uint_checked(0u128, dev_t_layout)?, // st_dev
            immty_from_uint_checked(mode, mode_t_layout)?, // st_mode
            immty_from_uint_checked(0u128, nlink_t_layout)?, // st_nlink
            immty_from_uint_checked(0u128, ino_t_layout)?, // st_ino
            immty_from_uint_checked(0u128, uid_t_layout)?, // st_uid
            immty_from_uint_checked(0u128, gid_t_layout)?, // st_gid
            immty_from_uint_checked(0u128, dev_t_layout)?, // st_rdev
            immty_from_uint_checked(0u128, uint32_t_layout)?, // padding
            immty_from_uint_checked(access_sec, time_t_layout)?, // st_atime
            immty_from_uint_checked(access_nsec, long_layout)?, // st_atime_nsec
            immty_from_uint_checked(modified_sec, time_t_layout)?, // st_mtime
            immty_from_uint_checked(modified_nsec, long_layout)?, // st_mtime_nsec
            immty_from_uint_checked(0u128, time_t_layout)?, // st_ctime
            immty_from_uint_checked(0u128, long_layout)?, // st_ctime_nsec
            immty_from_uint_checked(created_sec, time_t_layout)?, // st_birthtime
            immty_from_uint_checked(created_nsec, long_layout)?, // st_birthtime_nsec
            immty_from_uint_checked(metadata.size, off_t_layout)?, // st_size
            immty_from_uint_checked(0u128, blkcnt_t_layout)?, // st_blocks
            immty_from_uint_checked(0u128, blksize_t_layout)?, // st_blksize
            immty_from_uint_checked(0u128, uint32_t_layout)?, // st_flags
            immty_from_uint_checked(0u128, uint32_t_layout)?, // st_gen
        ];

        let buf = this.deref_operand(buf_op)?;
        this.write_packed_immediates(buf, &imms)?;

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

    fn file_type_to_d_type(&mut self, file_type: std::io::Result<FileType>) -> InterpResult<'tcx, i32> {
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
            Err(e) => return match e.raw_os_error() {
                Some(error) => Ok(error),
                None => throw_unsup_format!("the error {} couldn't be converted to a return value", e),
            }
        }
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
    streams: FxHashMap<u64, ReadDir>,
    /// ID number to be used by the next call to opendir
    next_id: u64,
}

impl DirHandler {
    fn insert_new(&mut self, read_dir: ReadDir) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.streams.insert(id, read_dir).unwrap_none();
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

fn maybe_sync_file(file: &File, writable: bool, operation: fn(&File) -> std::io::Result<()>) -> std::io::Result<i32> {
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
    fn open(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        flag_op: OpTy<'tcx, Tag>,
        mode_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("open")?;

        let flag = this.read_scalar(flag_op)?.to_i32()?;

        // Get the mode.  On macOS, the argument type `mode_t` is actually `u16`, but
        // C integer promotion rules mean that on the ABI level, it gets passed as `u32`
        // (see https://github.com/rust-lang/rust/issues/71915).
        let mode = this.read_scalar(mode_op)?.to_u32()?;
        if mode != 0o666 {
            throw_unsup_format!("non-default mode 0o{:o} is not supported", mode);
        }

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

        let path = this.read_path_from_c_str(this.read_scalar(path_op)?.check_init()?)?;

        let fd = options.open(&path).map(|file| {
            let fh = &mut this.machine.file_handler;
            fh.insert_fd(Box::new(FileHandle { file, writable }))
        });

        this.try_unwrap_io_result(fd)
    }

    fn fcntl(
        &mut self,
        args: &[OpTy<'tcx, Tag>],
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("fcntl")?;

        if args.len() < 2 {
            throw_ub_format!("incorrect number of arguments for fcntl: got {}, expected at least 2", args.len());
        }
        let fd = this.read_scalar(args[0])?.to_i32()?;
        let cmd = this.read_scalar(args[1])?.to_i32()?;
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
            let &[_, _, start] = check_arg_count(args)?;
            let start = this.read_scalar(start)?.to_i32()?;

            let fh = &mut this.machine.file_handler;

            match fh.handles.get_mut(&fd) {
                Some(file_descriptor) => {
                    let dup_result = file_descriptor.dup();
                    match dup_result {
                        Ok(dup_fd) => Ok(fh.insert_fd_with_min_fd(dup_fd, start)),
                        Err(e) => {
                            this.set_last_error_from_io_error(e)?;
                            Ok(-1)
                        }
                    }
                },
                None => return this.handle_not_found(),
            }
        } else if this.tcx.sess.target.target.target_os == "macos"
            && cmd == this.eval_libc_i32("F_FULLFSYNC")?
        {
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

    fn close(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("close")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        if let Some(file_descriptor) = this.machine.file_handler.handles.remove(&fd) {
            let result = file_descriptor.close(this.machine.communicate)?;
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn read(
        &mut self,
        fd: i32,
        buf: Scalar<Tag>,
        count: u64,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescriptor` trait.

        trace!("Reading from FD {}, size {}", fd, count);

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(
            buf,
            Size::from_bytes(count),
            Align::from_bytes(1).unwrap(),
        )?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.machine_isize_max() as u64).min(isize::MAX as u64);

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            trace!("read: FD mapped to {:?}", file_descriptor);
            // We want to read at most `count` bytes. We are sure that `count` is not negative
            // because it was a target's `usize`. Also we are sure that its smaller than
            // `usize::MAX` because it is a host's `isize`.
            let mut bytes = vec![0; count as usize];
            // `File::read` never returns a value larger than `count`,
            // so this cannot fail.
            let result = file_descriptor
                .read(this.machine.communicate, &mut bytes)?
                .map(|c| i64::try_from(c).unwrap());

            match result {
                Ok(read_bytes) => {
                    // If reading to `bytes` did not fail, we write those bytes to the buffer.
                    this.memory.write_bytes(buf, bytes)?;
                    Ok(read_bytes)
                }
                Err(e) => {
                    this.set_last_error_from_io_error(e)?;
                    Ok(-1)
                }
            }
        } else {
            trace!("read: FD not found");
            this.handle_not_found()
        }
    }

    fn write(
        &mut self,
        fd: i32,
        buf: Scalar<Tag>,
        count: u64,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        // Isolation check is done via `FileDescriptor` trait.

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(
            buf,
            Size::from_bytes(count),
            Align::from_bytes(1).unwrap(),
        )?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.machine_isize_max() as u64).min(isize::MAX as u64);

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            let bytes = this.memory.read_bytes(buf, Size::from_bytes(count))?;
            let result = file_descriptor
                .write(this.machine.communicate, &bytes)?
                .map(|c| i64::try_from(c).unwrap());
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn lseek64(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        offset_op: OpTy<'tcx, Tag>,
        whence_op: OpTy<'tcx, Tag>,
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

        if let Some(file_descriptor) = this.machine.file_handler.handles.get_mut(&fd) {
            let result = file_descriptor
                .seek(this.machine.communicate, seek_from)?
                .map(|offset| i64::try_from(offset).unwrap());
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn unlink(&mut self, path_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("unlink")?;

        let path = this.read_path_from_c_str(this.read_scalar(path_op)?.check_init()?)?;

        let result = remove_file(path).map(|_| 0);
        this.try_unwrap_io_result(result)
    }

    fn symlink(
        &mut self,
        target_op: OpTy<'tcx, Tag>,
        linkpath_op: OpTy<'tcx, Tag>
    ) -> InterpResult<'tcx, i32> {
        #[cfg(unix)]
        fn create_link(src: &Path, dst: &Path) -> std::io::Result<()> {
            std::os::unix::fs::symlink(src, dst)
        }

        #[cfg(windows)]
        fn create_link(src: &Path, dst: &Path) -> std::io::Result<()> {
            use std::os::windows::fs;
            if src.is_dir() {
                fs::symlink_dir(src, dst)
            } else {
                fs::symlink_file(src, dst)
            }
        }

        let this = self.eval_context_mut();

        this.check_no_isolation("symlink")?;

        let target = this.read_path_from_c_str(this.read_scalar(target_op)?.check_init()?)?;
        let linkpath = this.read_path_from_c_str(this.read_scalar(linkpath_op)?.check_init()?)?;

        let result = create_link(&target, &linkpath).map(|_| 0);
        this.try_unwrap_io_result(result)
    }

    fn macos_stat(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("macos", "stat");
        this.check_no_isolation("stat")?;
        // `stat` always follows symlinks.
        this.macos_stat_or_lstat(true, path_op, buf_op)
    }

    // `lstat` is used to get symlink metadata.
    fn macos_lstat(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("macos", "lstat");
        this.check_no_isolation("lstat")?;
        this.macos_stat_or_lstat(false, path_op, buf_op)
    }

    fn macos_fstat(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "fstat");
        this.check_no_isolation("fstat")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        let metadata = match FileMetadata::from_fd(this, fd)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };
        this.macos_stat_write_buf(metadata, buf_op)
    }

    fn linux_statx(
        &mut self,
        dirfd_op: OpTy<'tcx, Tag>,    // Should be an `int`
        pathname_op: OpTy<'tcx, Tag>, // Should be a `const char *`
        flags_op: OpTy<'tcx, Tag>,    // Should be an `int`
        _mask_op: OpTy<'tcx, Tag>,    // Should be an `unsigned int`
        statxbuf_op: OpTy<'tcx, Tag>, // Should be a `struct statx *`
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "statx");
        this.check_no_isolation("statx")?;

        let statxbuf_scalar = this.read_scalar(statxbuf_op)?.check_init()?;
        let pathname_scalar = this.read_scalar(pathname_op)?.check_init()?;

        // If the statxbuf or pathname pointers are null, the function fails with `EFAULT`.
        if this.is_null(statxbuf_scalar)? || this.is_null(pathname_scalar)? {
            let efault = this.eval_libc("EFAULT")?;
            this.set_last_error(efault)?;
            return Ok(-1);
        }

        // Under normal circumstances, we would use `deref_operand(statxbuf_op)` to produce a
        // proper `MemPlace` and then write the results of this function to it. However, the
        // `syscall` function is untyped. This means that all the `statx` parameters are provided
        // as `isize`s instead of having the proper types. Thus, we have to recover the layout of
        // `statxbuf_op` by using the `libc::statx` struct type.
        let statxbuf_place = {
            // FIXME: This long path is required because `libc::statx` is an struct and also a
            // function and `resolve_path` is returning the latter.
            let statx_ty = this
                .resolve_path(&["libc", "unix", "linux_like", "linux", "gnu", "statx"])
                .ty(*this.tcx, ty::ParamEnv::reveal_all());
            let statxbuf_ty = this.tcx.mk_mut_ptr(statx_ty);
            let statxbuf_layout = this.layout_of(statxbuf_ty)?;
            let statxbuf_imm = ImmTy::from_scalar(statxbuf_scalar, statxbuf_layout);
            this.ref_to_mplace(statxbuf_imm)?
        };

        let path = this.read_path_from_c_str(pathname_scalar)?.into_owned();
        // `flags` should be a `c_int` but the `syscall` function provides an `isize`.
        let flags: i32 =
            this.read_scalar(flags_op)?.to_machine_isize(&*this.tcx)?.try_into().map_err(|e| {
                err_unsup_format!("failed to convert pointer sized operand to integer: {}", e)
            })?;
        let empty_path_flag = flags & this.eval_libc("AT_EMPTY_PATH")?.to_i32()? != 0;
        // `dirfd` should be a `c_int` but the `syscall` function provides an `isize`.
        let dirfd: i32 =
            this.read_scalar(dirfd_op)?.to_machine_isize(&*this.tcx)?.try_into().map_err(|e| {
                err_unsup_format!("failed to convert pointer sized operand to integer: {}", e)
            })?;
        // We only support:
        // * interpreting `path` as an absolute directory,
        // * interpreting `path` as a path relative to `dirfd` when the latter is `AT_FDCWD`, or
        // * interpreting `dirfd` as any file descriptor when `path` is empty and AT_EMPTY_PATH is
        // set.
        // Other behaviors cannot be tested from `libstd` and thus are not implemented. If you
        // found this error, please open an issue reporting it.
        if !(
            path.is_absolute() ||
            dirfd == this.eval_libc_i32("AT_FDCWD")? ||
            (path.as_os_str().is_empty() && empty_path_flag)
        ) {
            throw_unsup_format!(
                "using statx is only supported with absolute paths, relative paths with the file \
                descriptor `AT_FDCWD`, and empty paths with the `AT_EMPTY_PATH` flag set and any \
                file descriptor"
            )
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
        let (access_sec, access_nsec) = metadata.accessed.map(|tup| {
            mask |= this.eval_libc("STATX_ATIME")?.to_u32()?;
            InterpResult::Ok(tup)
        }).unwrap_or(Ok((0, 0)))?;

        let (created_sec, created_nsec) = metadata.created.map(|tup| {
            mask |= this.eval_libc("STATX_BTIME")?.to_u32()?;
            InterpResult::Ok(tup)
        }).unwrap_or(Ok((0, 0)))?;

        let (modified_sec, modified_nsec) = metadata.modified.map(|tup| {
            mask |= this.eval_libc("STATX_MTIME")?.to_u32()?;
            InterpResult::Ok(tup)
        }).unwrap_or(Ok((0, 0)))?;

        let __u32_layout = this.libc_ty_layout("__u32")?;
        let __u64_layout = this.libc_ty_layout("__u64")?;
        let __u16_layout = this.libc_ty_layout("__u16")?;

        // Now we transform all this fields into `ImmTy`s and write them to `statxbuf`. We write a
        // zero for the unavailable fields.
        let imms = [
            immty_from_uint_checked(mask, __u32_layout)?, // stx_mask
            immty_from_uint_checked(0u128, __u32_layout)?, // stx_blksize
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_attributes
            immty_from_uint_checked(0u128, __u32_layout)?, // stx_nlink
            immty_from_uint_checked(0u128, __u32_layout)?, // stx_uid
            immty_from_uint_checked(0u128, __u32_layout)?, // stx_gid
            immty_from_uint_checked(mode, __u16_layout)?, // stx_mode
            immty_from_uint_checked(0u128, __u16_layout)?, // statx padding
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_ino
            immty_from_uint_checked(metadata.size, __u64_layout)?, // stx_size
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_blocks
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_attributes
            immty_from_uint_checked(access_sec, __u64_layout)?, // stx_atime.tv_sec
            immty_from_uint_checked(access_nsec, __u32_layout)?, // stx_atime.tv_nsec
            immty_from_uint_checked(0u128, __u32_layout)?, // statx_timestamp padding
            immty_from_uint_checked(created_sec, __u64_layout)?, // stx_btime.tv_sec
            immty_from_uint_checked(created_nsec, __u32_layout)?, // stx_btime.tv_nsec
            immty_from_uint_checked(0u128, __u32_layout)?, // statx_timestamp padding
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_ctime.tv_sec
            immty_from_uint_checked(0u128, __u32_layout)?, // stx_ctime.tv_nsec
            immty_from_uint_checked(0u128, __u32_layout)?, // statx_timestamp padding
            immty_from_uint_checked(modified_sec, __u64_layout)?, // stx_mtime.tv_sec
            immty_from_uint_checked(modified_nsec, __u32_layout)?, // stx_mtime.tv_nsec
            immty_from_uint_checked(0u128, __u32_layout)?, // statx_timestamp padding
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_rdev_major
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_rdev_minor
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_dev_major
            immty_from_uint_checked(0u128, __u64_layout)?, // stx_dev_minor
        ];

        this.write_packed_immediates(statxbuf_place, &imms)?;

        Ok(0)
    }

    fn rename(
        &mut self,
        oldpath_op: OpTy<'tcx, Tag>,
        newpath_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("rename")?;

        let oldpath_scalar = this.read_scalar(oldpath_op)?.check_init()?;
        let newpath_scalar = this.read_scalar(newpath_op)?.check_init()?;

        if this.is_null(oldpath_scalar)? || this.is_null(newpath_scalar)? {
            let efault = this.eval_libc("EFAULT")?;
            this.set_last_error(efault)?;
            return Ok(-1);
        }

        let oldpath = this.read_path_from_c_str(oldpath_scalar)?;
        let newpath = this.read_path_from_c_str(newpath_scalar)?;

        let result = rename(oldpath, newpath).map(|_| 0);

        this.try_unwrap_io_result(result)
    }

    fn mkdir(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        mode_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("mkdir")?;

        #[cfg_attr(not(unix), allow(unused_variables))]
        let mode = if this.tcx.sess.target.target.target_os == "macos" {
            u32::from(this.read_scalar(mode_op)?.check_init()?.to_u16()?)
        } else {
            this.read_scalar(mode_op)?.to_u32()?
        };

        let path = this.read_path_from_c_str(this.read_scalar(path_op)?.check_init()?)?;

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

    fn rmdir(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("rmdir")?;

        let path = this.read_path_from_c_str(this.read_scalar(path_op)?.check_init()?)?;

        let result = remove_dir(path).map(|_| 0i32);

        this.try_unwrap_io_result(result)
    }

    fn opendir(&mut self, name_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        this.check_no_isolation("opendir")?;

        let name = this.read_path_from_c_str(this.read_scalar(name_op)?.check_init()?)?;

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
                this.set_last_error_from_io_error(e)?;
                Ok(Scalar::null_ptr(this))
            }
        }
    }

    fn linux_readdir64_r(
        &mut self,
        dirp_op: OpTy<'tcx, Tag>,
        entry_op: OpTy<'tcx, Tag>,
        result_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "readdir64_r");
        this.check_no_isolation("readdir64_r")?;

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        let dir_iter = this.machine.dir_handler.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir64_r did not come from opendir")
        })?;
        match dir_iter.next() {
            Some(Ok(dir_entry)) => {
                // Write into entry, write pointer to result, return 0 on success.
                // The name is written with write_os_str_to_c_str, while the rest of the
                // dirent64 struct is written using write_packed_immediates.

                // For reference:
                // pub struct dirent64 {
                //     pub d_ino: ino64_t,
                //     pub d_off: off64_t,
                //     pub d_reclen: c_ushort,
                //     pub d_type: c_uchar,
                //     pub d_name: [c_char; 256],
                // }

                let entry_place = this.deref_operand(entry_op)?;
                let name_place = this.mplace_field(entry_place, 4)?;

                let file_name = dir_entry.file_name(); // not a Path as there are no separators!
                let (name_fits, _) = this.write_os_str_to_c_str(
                    &file_name,
                    name_place.ptr,
                    name_place.layout.size.bytes(),
                )?;
                if !name_fits {
                    throw_unsup_format!("a directory entry had a name too large to fit in libc::dirent64");
                }

                let entry_place = this.deref_operand(entry_op)?;
                let ino64_t_layout = this.libc_ty_layout("ino64_t")?;
                let off64_t_layout = this.libc_ty_layout("off64_t")?;
                let c_ushort_layout = this.libc_ty_layout("c_ushort")?;
                let c_uchar_layout = this.libc_ty_layout("c_uchar")?;

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;

                let imms = [
                    immty_from_uint_checked(ino, ino64_t_layout)?, // d_ino
                    immty_from_uint_checked(0u128, off64_t_layout)?, // d_off
                    immty_from_uint_checked(0u128, c_ushort_layout)?, // d_reclen
                    immty_from_int_checked(file_type, c_uchar_layout)?, // d_type
                ];
                this.write_packed_immediates(entry_place, &imms)?;

                let result_place = this.deref_operand(result_op)?;
                this.write_scalar(this.read_scalar(entry_op)?, result_place.into())?;

                Ok(0)
            }
            None => {
                // end of stream: return 0, assign *result=NULL
                this.write_null(this.deref_operand(result_op)?.into())?;
                Ok(0)
            }
            Some(Err(e)) => match e.raw_os_error() {
                // return positive error number on error
                Some(error) => Ok(error),
                None => {
                    throw_unsup_format!("the error {} couldn't be converted to a return value", e)
                }
            },
        }
    }

    fn macos_readdir_r(
        &mut self,
        dirp_op: OpTy<'tcx, Tag>,
        entry_op: OpTy<'tcx, Tag>,
        result_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "readdir_r");
        this.check_no_isolation("readdir_r")?;

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        let dir_iter = this.machine.dir_handler.streams.get_mut(&dirp).ok_or_else(|| {
            err_unsup_format!("the DIR pointer passed to readdir_r did not come from opendir")
        })?;
        match dir_iter.next() {
            Some(Ok(dir_entry)) => {
                // Write into entry, write pointer to result, return 0 on success.
                // The name is written with write_os_str_to_c_str, while the rest of the
                // dirent struct is written using write_packed_Immediates.

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
                let name_place = this.mplace_field(entry_place, 5)?;

                let file_name = dir_entry.file_name(); // not a Path as there are no separators!
                let (name_fits, file_name_len) = this.write_os_str_to_c_str(
                    &file_name,
                    name_place.ptr,
                    name_place.layout.size.bytes(),
                )?;
                if !name_fits {
                    throw_unsup_format!("a directory entry had a name too large to fit in libc::dirent");
                }

                let entry_place = this.deref_operand(entry_op)?;
                let ino_t_layout = this.libc_ty_layout("ino_t")?;
                let off_t_layout = this.libc_ty_layout("off_t")?;
                let c_ushort_layout = this.libc_ty_layout("c_ushort")?;
                let c_uchar_layout = this.libc_ty_layout("c_uchar")?;

                // If the host is a Unix system, fill in the inode number with its real value.
                // If not, use 0 as a fallback value.
                #[cfg(unix)]
                let ino = std::os::unix::fs::DirEntryExt::ino(&dir_entry);
                #[cfg(not(unix))]
                let ino = 0u64;

                let file_type = this.file_type_to_d_type(dir_entry.file_type())?;

                let imms = [
                    immty_from_uint_checked(ino, ino_t_layout)?, // d_ino
                    immty_from_uint_checked(0u128, off_t_layout)?, // d_seekoff
                    immty_from_uint_checked(0u128, c_ushort_layout)?, // d_reclen
                    immty_from_uint_checked(file_name_len, c_ushort_layout)?, // d_namlen
                    immty_from_int_checked(file_type, c_uchar_layout)?, // d_type
                ];
                this.write_packed_immediates(entry_place, &imms)?;

                let result_place = this.deref_operand(result_op)?;
                this.write_scalar(this.read_scalar(entry_op)?, result_place.into())?;

                Ok(0)
            }
            None => {
                // end of stream: return 0, assign *result=NULL
                this.write_null(this.deref_operand(result_op)?.into())?;
                Ok(0)
            }
            Some(Err(e)) => match e.raw_os_error() {
                // return positive error number on error
                Some(error) => Ok(error),
                None => {
                    throw_unsup_format!("the error {} couldn't be converted to a return value", e)
                }
            },
        }
    }

    fn closedir(&mut self, dirp_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("closedir")?;

        let dirp = this.read_scalar(dirp_op)?.to_machine_usize(this)?;

        if let Some(dir_iter) = this.machine.dir_handler.streams.remove(&dirp) {
            drop(dir_iter);
            Ok(0)
        } else {
            this.handle_not_found()
        }
    }

    fn ftruncate64(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        length_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("ftruncate64")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let length = this.read_scalar(length_op)?.to_i64()?;
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

    fn fsync(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        // On macOS, `fsync` (unlike `fcntl(F_FULLFSYNC)`) does not wait for the
        // underlying disk to finish writing. In the interest of host compatibility,
        // we conservatively implement this with `sync_all`, which
        // *does* wait for the disk.

        let this = self.eval_context_mut();

        this.check_no_isolation("fsync")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
            // FIXME: Support fsync for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            let io_result = maybe_sync_file(&file, *writable, File::sync_all);
            this.try_unwrap_io_result(io_result)
        } else {
            this.handle_not_found()
        }
    }

    fn fdatasync(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("fdatasync")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
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
        fd_op: OpTy<'tcx, Tag>,
        offset_op: OpTy<'tcx, Tag>,
        nbytes_op: OpTy<'tcx, Tag>,
        flags_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("sync_file_range")?;

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

        if let Some(file_descriptor) = this.machine.file_handler.handles.get(&fd) {
            // FIXME: Support sync_data_range for all FDs
            let FileHandle { file, writable } = file_descriptor.as_file_handle()?;
            let io_result = maybe_sync_file(&file, *writable, File::sync_data);
            this.try_unwrap_io_result(io_result)
        } else {
            this.handle_not_found()
        }
    }
}

/// Extracts the number of seconds and nanoseconds elapsed between `time` and the unix epoch when
/// `time` is Ok. Returns `None` if `time` is an error. Fails if `time` happens before the unix
/// epoch.
fn extract_sec_and_nsec<'tcx>(
    time: std::io::Result<SystemTime>
) -> InterpResult<'tcx, Option<(u64, u32)>> {
    time.ok().map(|time| {
        let duration = system_time_to_duration(&time)?;
        Ok((duration.as_secs(), duration.subsec_nanos()))
    }).transpose()
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
        follow_symlink: bool
    ) -> InterpResult<'tcx, Option<FileMetadata>> {
        let metadata = if follow_symlink {
            std::fs::metadata(path)
        } else {
            std::fs::symlink_metadata(path)
        };

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
                ecx.set_last_error_from_io_error(e)?;
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
