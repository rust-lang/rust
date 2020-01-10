use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fs::{remove_file, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::SystemTime;

use rustc::ty::layout::{Align, LayoutOf, Size};

use crate::stacked_borrows::Tag;
use crate::*;
use helpers::immty_from_uint_checked;
use shims::time::system_time_to_duration;

#[derive(Debug)]
pub struct FileHandle {
    file: File,
    writable: bool,
}

pub struct FileHandler {
    handles: HashMap<i32, FileHandle>,
    low: i32,
}

impl Default for FileHandler {
    fn default() -> Self {
        FileHandler {
            handles: Default::default(),
            // 0, 1 and 2 are reserved for stdin, stdout and stderr.
            low: 3,
        }
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn open(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        flag_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("open")?;

        let flag = this.read_scalar(flag_op)?.to_i32()?;

        let mut options = OpenOptions::new();

        let o_rdonly = this.eval_libc_i32("O_RDONLY")?;
        let o_wronly = this.eval_libc_i32("O_WRONLY")?;
        let o_rdwr = this.eval_libc_i32("O_RDWR")?;
        // The first two bits of the flag correspond to the access mode in linux, macOS and
        // windows. We need to check that in fact the access mode flags for the current platform
        // only use these two bits, otherwise we are in an unsupported platform and should error.
        if (o_rdonly | o_wronly | o_rdwr) & !0b11 != 0 {
            throw_unsup_format!("Access mode flags on this platform are unsupported");
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
            throw_unsup_format!("Unsupported access mode {:#x}", access_mode);
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
            options.create(true);
            mirror |= o_creat;
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

        let path = this.read_os_str_from_c_str(this.read_scalar(path_op)?.not_undef()?)?;

        let fd = options.open(&path).map(|file| {
            let mut fh = &mut this.machine.file_handler;
            fh.low += 1;
            fh.handles.insert(fh.low, FileHandle { file, writable }).unwrap_none();
            fh.low
        });

        this.try_unwrap_io_result(fd)
    }

    fn fcntl(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        cmd_op: OpTy<'tcx, Tag>,
        _arg1_op: Option<OpTy<'tcx, Tag>>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("fcntl")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let cmd = this.read_scalar(cmd_op)?.to_i32()?;
        // We only support getting the flags for a descriptor.
        if cmd == this.eval_libc_i32("F_GETFD")? {
            // Currently this is the only flag that `F_GETFD` returns. It is OK to just return the
            // `FD_CLOEXEC` value without checking if the flag is set for the file because `std`
            // always sets this flag when opening a file. However we still need to check that the
            // file itself is open.
            if this.machine.file_handler.handles.contains_key(&fd) {
                Ok(this.eval_libc_i32("FD_CLOEXEC")?)
            } else {
                this.handle_not_found()
            }
        } else {
            throw_unsup_format!("The {:#x} command is not supported for `fcntl`)", cmd);
        }
    }

    fn close(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("close")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        if let Some(handle) = this.machine.file_handler.handles.remove(&fd) {
            // We sync the file if it was opened in a mode different than read-only.
            if handle.writable {
                // `File::sync_all` does the checks that are done when closing a file. We do this to
                // to handle possible errors correctly.
                let result = this.try_unwrap_io_result(handle.file.sync_all().map(|_| 0i32));
                // Now we actually close the file.
                drop(handle);
                // And return the result.
                result
            } else {
                // We drop the file, this closes it but ignores any errors produced when closing
                // it. This is done because `File::sync_call` cannot be done over files like
                // `/dev/urandom` which are read-only. Check
                // https://github.com/rust-lang/miri/issues/999#issuecomment-568920439 for a deeper
                // discussion.
                drop(handle);
                Ok(0)
            }
        } else {
            this.handle_not_found()
        }
    }

    fn read(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
        count_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        this.check_no_isolation("read")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let buf = this.read_scalar(buf_op)?.not_undef()?;
        let count = this.read_scalar(count_op)?.to_machine_usize(&*this.tcx)?;

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(
            buf,
            Size::from_bytes(count),
            Align::from_bytes(1).unwrap(),
        )?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.isize_max() as u64).min(isize::max_value() as u64);

        if let Some(handle) = this.machine.file_handler.handles.get_mut(&fd) {
            // This can never fail because `count` was capped to be smaller than
            // `isize::max_value()`.
            let count = isize::try_from(count).unwrap();
            // We want to read at most `count` bytes. We are sure that `count` is not negative
            // because it was a target's `usize`. Also we are sure that its smaller than
            // `usize::max_value()` because it is a host's `isize`.
            let mut bytes = vec![0; count as usize];
            let result = handle
                .file
                .read(&mut bytes)
                // `File::read` never returns a value larger than `count`, so this cannot fail.
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
            this.handle_not_found()
        }
    }

    fn write(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
        count_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        this.check_no_isolation("write")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let buf = this.read_scalar(buf_op)?.not_undef()?;
        let count = this.read_scalar(count_op)?.to_machine_usize(&*this.tcx)?;

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(
            buf,
            Size::from_bytes(count),
            Align::from_bytes(1).unwrap(),
        )?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count.min(this.isize_max() as u64).min(isize::max_value() as u64);

        if let Some(handle) = this.machine.file_handler.handles.get_mut(&fd) {
            let bytes = this.memory.read_bytes(buf, Size::from_bytes(count))?;
            let result = handle.file.write(&bytes).map(|c| i64::try_from(c).unwrap());
            this.try_unwrap_io_result(result)
        } else {
            this.handle_not_found()
        }
    }

    fn unlink(&mut self, path_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("unlink")?;

        let path = this.read_os_str_from_c_str(this.read_scalar(path_op)?.not_undef()?)?;

        let result = remove_file(path).map(|_| 0);

        this.try_unwrap_io_result(result)
    }

    fn symlink(
        &mut self,
        target_op: OpTy<'tcx, Tag>,
        linkpath_op: OpTy<'tcx, Tag>
    ) -> InterpResult<'tcx, i32> {
        #[cfg(target_family = "unix")]
        fn create_link(src: PathBuf, dst: PathBuf) -> std::io::Result<()> {
            std::os::unix::fs::symlink(src, dst)
        }

        #[cfg(target_family = "windows")]
        fn create_link(src: PathBuf, dst: PathBuf) -> std::io::Result<()> {
            use std::os::windows::fs;
            if src.is_dir() {
                fs::symlink_dir(src, dst)
            } else {
                fs::symlink_file(src, dst)
            }
        }

        let this = self.eval_context_mut();

        this.check_no_isolation("symlink")?;

        let target = this.read_os_str_from_c_str(this.read_scalar(target_op)?.not_undef()?)?.into();
        let linkpath = this.read_os_str_from_c_str(this.read_scalar(linkpath_op)?.not_undef()?)?.into();

        this.try_unwrap_io_result(create_link(target, linkpath).map(|_| 0))
    }

    fn stat(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.check_no_isolation("stat")?;
        // `stat` always follows symlinks.
        this.stat_or_lstat(true, path_op, buf_op)
    }

    // `lstat` is used to get symlink metadata.
    fn lstat(
        &mut self,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.check_no_isolation("lstat")?;
        this.stat_or_lstat(false, path_op, buf_op)
    }

    fn stat_or_lstat(
        &mut self,
        follow_symlink: bool,
        path_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if this.tcx.sess.target.target.target_os.to_lowercase() != "macos" {
            throw_unsup_format!("The `stat` and `lstat` shims are only available for `macos` targets.")
        }

        let path_scalar = this.read_scalar(path_op)?.not_undef()?;
        let path: PathBuf = this.read_os_str_from_c_str(path_scalar)?.into();

        let buf = this.deref_operand(buf_op)?;

        let metadata = match FileMetadata::new(this, path, follow_symlink)? {
            Some(metadata) => metadata,
            None => return Ok(-1),
        };

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

        // We need to add 32 bits of padding after `st_rdev` if we are on a 64-bit platform.
        let pad_layout = if this.tcx.sess.target.ptr_width == 64 {
            uint32_t_layout
        } else {
            this.layout_of(this.tcx.mk_unit())?
        };

        let imms = [
            immty_from_uint_checked(0u128, dev_t_layout)?, // st_dev
            immty_from_uint_checked(mode, mode_t_layout)?, // st_mode
            immty_from_uint_checked(0u128, nlink_t_layout)?, // st_nlink
            immty_from_uint_checked(0u128, ino_t_layout)?, // st_ino
            immty_from_uint_checked(0u128, uid_t_layout)?, // st_uid
            immty_from_uint_checked(0u128, gid_t_layout)?, // st_gid
            immty_from_uint_checked(0u128, dev_t_layout)?, // st_rdev
            immty_from_uint_checked(0u128, pad_layout)?, // padding for 64-bit targets
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

        this.write_packed_immediates(&buf, &imms)?;

        Ok(0)
    }

    fn statx(
        &mut self,
        dirfd_op: OpTy<'tcx, Tag>,    // Should be an `int`
        pathname_op: OpTy<'tcx, Tag>, // Should be a `const char *`
        flags_op: OpTy<'tcx, Tag>,    // Should be an `int`
        _mask_op: OpTy<'tcx, Tag>,    // Should be an `unsigned int`
        statxbuf_op: OpTy<'tcx, Tag>, // Should be a `struct statx *`
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("statx")?;

        if this.tcx.sess.target.target.target_os.to_lowercase() != "linux" {
            throw_unsup_format!("The `statx` shim is only available for `linux` targets.")
        }

        let statxbuf_scalar = this.read_scalar(statxbuf_op)?.not_undef()?;
        let pathname_scalar = this.read_scalar(pathname_op)?.not_undef()?;

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
                .resolve_path(&["libc", "unix", "linux_like", "linux", "gnu", "statx"])?
                .monomorphic_ty(*this.tcx);
            let statxbuf_ty = this.tcx.mk_mut_ptr(statx_ty);
            let statxbuf_layout = this.layout_of(statxbuf_ty)?;
            let statxbuf_imm = ImmTy::from_scalar(statxbuf_scalar, statxbuf_layout);
            this.ref_to_mplace(statxbuf_imm)?
        };

        let path: PathBuf = this.read_os_str_from_c_str(pathname_scalar)?.into();
        // `flags` should be a `c_int` but the `syscall` function provides an `isize`.
        let flags: i32 =
            this.read_scalar(flags_op)?.to_machine_isize(&*this.tcx)?.try_into().map_err(|e| {
                err_unsup_format!("Failed to convert pointer sized operand to integer: {}", e)
            })?;
        // `dirfd` should be a `c_int` but the `syscall` function provides an `isize`.
        let dirfd: i32 =
            this.read_scalar(dirfd_op)?.to_machine_isize(&*this.tcx)?.try_into().map_err(|e| {
                err_unsup_format!("Failed to convert pointer sized operand to integer: {}", e)
            })?;
        // we only support interpreting `path` as an absolute directory or as a directory relative
        // to `dirfd` when the latter is `AT_FDCWD`. The behavior of `statx` with a relative path
        // and a directory file descriptor other than `AT_FDCWD` is specified but it cannot be
        // tested from `libstd`. If you found this error, please open an issue reporting it.
        if !(path.is_absolute() || dirfd == this.eval_libc_i32("AT_FDCWD")?) {
            throw_unsup_format!(
                "Using statx with a relative path and a file descriptor different from `AT_FDCWD` is not supported"
            )
        }

        // the `_mask_op` paramter specifies the file information that the caller requested.
        // However `statx` is allowed to return information that was not requested or to not
        // return information that was requested. This `mask` represents the information we can
        // actually provide in any host platform.
        let mut mask =
            this.eval_libc("STATX_TYPE")?.to_u32()? | this.eval_libc("STATX_SIZE")?.to_u32()?;

        // If the `AT_SYMLINK_NOFOLLOW` flag is set, we query the file's metadata without following
        // symbolic links.
        let follow_symlink = flags & this.eval_libc("AT_SYMLINK_NOFOLLOW")?.to_i32()? == 0;

        let metadata = match FileMetadata::new(this, path, follow_symlink)? {
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

        this.write_packed_immediates(&statxbuf_place, &imms)?;

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
    fn new<'tcx, 'mir>(
        ecx: &mut MiriEvalContext<'mir, 'tcx>,
        path: PathBuf,
        follow_symlink: bool
    ) -> InterpResult<'tcx, Option<FileMetadata>> {
        let metadata = if follow_symlink {
            std::fs::metadata(path)
        } else {
            std::fs::symlink_metadata(path)
        };

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
