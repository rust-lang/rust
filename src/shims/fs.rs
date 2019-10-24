use std::collections::HashMap;
use std::fs::{File, OpenOptions, remove_file};
use std::io::{Read, Write};

use rustc::ty::layout::Size;

use crate::stacked_borrows::Tag;
use crate::*;

#[derive(Debug)]
pub struct FileHandle {
    file: File,
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
        // Now we check the access mode
        let access_mode = flag & 0b11;

        if access_mode == o_rdonly {
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

        let path = this.read_os_string_from_c_string(this.read_scalar(path_op)?.not_undef()?)?;

        let fd = options.open(path).map(|file| {
            let mut fh = &mut this.machine.file_handler;
            fh.low += 1;
            fh.handles.insert(fh.low, FileHandle { file }).unwrap_none();
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
            let fd_cloexec = this.eval_libc_i32("FD_CLOEXEC")?;
            this.get_handle_and(fd, |_| Ok(fd_cloexec))
        } else {
            throw_unsup_format!("The {:#x} command is not supported for `fcntl`)", cmd);
        }
    }

    fn close(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("close")?;

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        this.remove_handle_and(fd, |handle, this| {
            this.try_unwrap_io_result(handle.file.sync_all().map(|_| 0i32))
        })
    }

    fn read(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
        count_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        this.check_no_isolation("read")?;

        let count = this.read_scalar(count_op)?.to_usize(&*this.tcx)?;
        // Reading zero bytes should not change `buf`.
        if count == 0 {
            return Ok(0);
        }
        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let buf_scalar = this.read_scalar(buf_op)?.not_undef()?;

        // Remove the file handle to avoid borrowing issues.
        this.remove_handle_and(fd, |mut handle, this| {
            // Don't use `?` to avoid returning before reinserting the handle.
            let bytes = this.force_ptr(buf_scalar).and_then(|buf| {
                this.memory
                    .get_mut(buf.alloc_id)?
                    .get_bytes_mut(&*this.tcx, buf, Size::from_bytes(count))
                    .map(|buffer| handle.file.read(buffer))
            });
            this.machine.file_handler.handles.insert(fd, handle).unwrap_none();
            this.try_unwrap_io_result(bytes?.map(|bytes| bytes as i64))
        })
    }

    fn write(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
        count_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        this.check_no_isolation("write")?;

        let count = this.read_scalar(count_op)?.to_usize(&*this.tcx)?;
        // Writing zero bytes should not change `buf`.
        if count == 0 {
            return Ok(0);
        }
        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let buf = this.force_ptr(this.read_scalar(buf_op)?.not_undef()?)?;

        this.remove_handle_and(fd, |mut handle, this| {
            let bytes = this.memory.get(buf.alloc_id).and_then(|alloc| {
                alloc
                    .get_bytes(&*this.tcx, buf, Size::from_bytes(count))
                    .map(|bytes| handle.file.write(bytes).map(|bytes| bytes as i64))
            });
            this.machine.file_handler.handles.insert(fd, handle).unwrap_none();
            this.try_unwrap_io_result(bytes?)
        })
    }

    fn unlink( &mut self, path_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("unlink")?;

        let path = this.read_os_string_from_c_string(this.read_scalar(path_op)?.not_undef()?)?;

        let result = remove_file(path).map(|_| 0);

        this.try_unwrap_io_result(result)
    }

    /// Helper function that gets a `FileHandle` immutable reference and allows to manipulate it
    /// using the `f` closure.
    ///
    /// If the `fd` file descriptor does not correspond to a file, this functions returns `Ok(-1)`
    /// and sets `Evaluator::last_error` to `libc::EBADF` (invalid file descriptor).
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`).
    fn get_handle_and<F, T: From<i32>>(&mut self, fd: i32, f: F) -> InterpResult<'tcx, T>
    where
        F: Fn(&FileHandle) -> InterpResult<'tcx, T>,
    {
        let this = self.eval_context_mut();
        if let Some(handle) = this.machine.file_handler.handles.get(&fd) {
            f(handle)
        } else {
            let ebadf = this.eval_libc("EBADF")?;
            this.set_last_error(ebadf)?;
            Ok((-1).into())
        }
    }

    /// Helper function that removes a `FileHandle` and allows to manipulate it using the `f`
    /// closure. This function is quite useful when you need to modify a `FileHandle` but you need
    /// to modify `MiriEvalContext` at the same time, so you can modify the handle and reinsert it
    /// using `f`.
    ///
    /// If the `fd` file descriptor does not correspond to a file, this functions returns `Ok(-1)`
    /// and sets `Evaluator::last_error` to `libc::EBADF` (invalid file descriptor).
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`).
    fn remove_handle_and<F, T: From<i32>>(&mut self, fd: i32, mut f: F) -> InterpResult<'tcx, T>
    where
        F: FnMut(FileHandle, &mut MiriEvalContext<'mir, 'tcx>) -> InterpResult<'tcx, T>,
    {
        let this = self.eval_context_mut();
        if let Some(handle) = this.machine.file_handler.handles.remove(&fd) {
            f(handle, this)
        } else {
            let ebadf = this.eval_libc("EBADF")?;
            this.set_last_error(ebadf)?;
            Ok((-1).into())
        }
    }
}
