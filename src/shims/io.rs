use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use rustc::ty::layout::Size;

use crate::stacked_borrows::Tag;
use crate::*;

pub struct FileHandle {
    file: File,
    flag: i32,
}

pub struct FileHandler {
    handles: HashMap<i32, FileHandle>,
    low: i32,
}

impl Default for FileHandler {
    fn default() -> Self {
        FileHandler {
            handles: Default::default(),
            // 0, 1 and 2 are reserved for stdin, stdout and stderr
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

        if !this.machine.communicate {
            throw_unsup_format!("`open` not available when isolation is enabled")
        }

        let flag = this.read_scalar(flag_op)?.to_i32()?;

        if flag != this.eval_libc_i32("O_RDONLY")? && flag != this.eval_libc_i32("O_CLOEXEC")? {
            throw_unsup_format!("Unsupported flag {:#x}", flag);
        }

        let path_bytes = this
            .memory()
            .read_c_str(this.read_scalar(path_op)?.not_undef()?)?;
        let path = std::str::from_utf8(path_bytes)
            .map_err(|_| err_unsup_format!("{:?} is not a valid utf-8 string", path_bytes))?;
        let fd = File::open(path).map(|file| {
            let mut fh = &mut this.machine.file_handler;
            fh.low += 1;
            fh.handles.insert(fh.low, FileHandle { file, flag });
            fh.low
        });

        this.consume_result(fd)
    }

    fn fcntl(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        cmd_op: OpTy<'tcx, Tag>,
        arg_op: Option<OpTy<'tcx, Tag>>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.machine.communicate {
            throw_unsup_format!("`open` not available when isolation is enabled")
        }

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let cmd = this.read_scalar(cmd_op)?.to_i32()?;

        if cmd == this.eval_libc_i32("F_SETFD")? {
            // This does not affect the file itself. Certain flags might require changing the file
            // or the way it is accessed somehow.
            let flag = this.read_scalar(arg_op.unwrap())?.to_i32()?;
            // The only usage of this in stdlib at the moment is to enable the `FD_CLOEXEC` flag.
            let fd_cloexec = this.eval_libc_i32("FD_CLOEXEC")?;
            if let Some(FileHandle { flag: old_flag, .. }) =
                this.machine.file_handler.handles.get_mut(&fd)
            {
                if flag ^ *old_flag == fd_cloexec {
                    *old_flag = flag;
                } else {
                    throw_unsup_format!("Unsupported arg {:#x} for `F_SETFD`", flag);
                }
            }
            Ok(0)
        } else if cmd == this.eval_libc_i32("F_GETFD")? {
            this.get_handle_and(fd, |handle| Ok(handle.flag))
        } else {
            throw_unsup_format!("Unsupported command {:#x}", cmd);
        }
    }

    fn close(&mut self, fd_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.machine.communicate {
            throw_unsup_format!("`open` not available when isolation is enabled")
        }

        let fd = this.read_scalar(fd_op)?.to_i32()?;

        this.remove_handle_and(
            fd,
            |handle, this| this.consume_result(handle.file.sync_all().map(|_| 0i32)),
        )
    }

    fn read(
        &mut self,
        fd_op: OpTy<'tcx, Tag>,
        buf_op: OpTy<'tcx, Tag>,
        count_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i64> {
        let this = self.eval_context_mut();

        if !this.machine.communicate {
            throw_unsup_format!("`open` not available when isolation is enabled")
        }

        let tcx = &{ this.tcx.tcx };

        let fd = this.read_scalar(fd_op)?.to_i32()?;
        let buf = this.force_ptr(this.read_scalar(buf_op)?.not_undef()?)?;
        let count = this.read_scalar(count_op)?.to_usize(&*this.tcx)?;

        // Remove the file handle to avoid borrowing issues
        this.remove_handle_and(
            fd,
            |mut handle, this| {
                let bytes = handle
                    .file
                    .read(this.memory_mut().get_mut(buf.alloc_id)?.get_bytes_mut(
                        tcx,
                        buf,
                        Size::from_bytes(count),
                    )?)
                    .map(|bytes| bytes as i64);
                // Reinsert the file handle
                this.machine.file_handler.handles.insert(fd, handle);
                this.consume_result(bytes)
            },
        )
    }

    /// Helper function that gets a `FileHandle` immutable reference and allows to manipulate it
    /// using `f`.
    ///
    /// If the `fd` file descriptor does not corresponds to a file, this functions returns `Ok(-1)`
    /// and sets `Evaluator::last_error` to `libc::EBADF` (invalid file descriptor).
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`)
    fn get_handle_and<F, T: From<i32>>(&mut self, fd: i32, f: F) -> InterpResult<'tcx, T>
    where
        F: Fn(&FileHandle) -> InterpResult<'tcx, T>,
    {
        let this = self.eval_context_mut();
        if let Some(handle) = this.machine.file_handler.handles.get(&fd) {
            f(handle)
        } else {
            this.machine.last_error = this.eval_libc_i32("EBADF")? as u32;
            Ok((-1).into())
        }
    }

    /// Helper function that removes a `FileHandle` and allows to manipulate it using the `f`
    /// closure. This function is quite useful when you need to modify a `FileHandle` but you need
    /// to modify `MiriEvalContext` at the same time, so you can modify the handle and reinsert it
    /// using `f`.
    ///
    /// If the `fd` file descriptor does not corresponds to a file, this functions returns `Ok(-1)`
    /// and sets `Evaluator::last_error` to `libc::EBADF` (invalid file descriptor).
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`)
    fn remove_handle_and<F, T: From<i32>>(&mut self, fd: i32, mut f: F) -> InterpResult<'tcx, T>
    where
        F: FnMut(FileHandle, &mut MiriEvalContext<'mir, 'tcx>) -> InterpResult<'tcx, T>,
    {
        let this = self.eval_context_mut();
        if let Some(handle) = this.machine.file_handler.handles.remove(&fd) {
            f(handle, this)
        } else {
            this.machine.last_error = this.eval_libc_i32("EBADF")? as u32;
            Ok((-1).into())
        }
    }

    /// Helper function that consumes an `std::io::Result<T>` and returns an
    /// `InterpResult<'tcx,T>::Ok` instead. It is expected that the result can be converted to an
    /// OS error using `std::io::Error::raw_os_error`.
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`)
    fn consume_result<T: From<i32>>(&mut self, result: std::io::Result<T>) -> InterpResult<'tcx, T> {
        match result {
            Ok(ok) => Ok(ok),
            Err(e) => {
                self.eval_context_mut().machine.last_error = e.raw_os_error().unwrap() as u32;
                Ok((-1).into())
            }
        }
    }
}
