use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{Read, Write};

use rustc::ty::layout::{Size, Align};

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

        let path = this.read_os_str_from_c_str(this.read_scalar(path_op)?.not_undef()?)?;

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
            // `File::sync_all` does the checks that are done when closing a file. We do this to
            // to handle possible errors correctly.
            let result = this.try_unwrap_io_result(handle.file.sync_all().map(|_| 0i32));
            // Now we actually close the file.
            drop(handle);
            // And return the result.
            result
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
        let count = this
            .read_scalar(count_op)?
            .to_machine_usize(&*this.tcx)?;

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(buf, Size::from_bytes(count), Align::from_bytes(1).unwrap())?;

        // We cap the number of read bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(this.isize_max() as u64)
            .min(isize::max_value() as u64);

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
        let count = this
            .read_scalar(count_op)?
            .to_machine_usize(&*this.tcx)?;

        // Check that the *entire* buffer is actually valid memory.
        this.memory.check_ptr_access(buf, Size::from_bytes(count), Align::from_bytes(1).unwrap())?;

        // We cap the number of written bytes to the largest value that we are able to fit in both the
        // host's and target's `isize`. This saves us from having to handle overflows later.
        let count = count
            .min(this.isize_max() as u64)
            .min(isize::max_value() as u64);

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
