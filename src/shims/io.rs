use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use crate::stacked_borrows::Tag;
use crate::*;

#[derive(Default)]
pub struct FileHandler {
    files: HashMap<i32, File>,
    flags: HashMap<i32, i32>,
    low: i32,
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

        match File::open(path) {
            Ok(file) => {
                let mut fh = &mut this.machine.file_handler;
                fh.low += 1;
                fh.files.insert(fh.low, file);
                fh.flags.insert(fh.low, flag);
                Ok(fh.low)
            }

            Err(e) => {
                this.machine.last_error = e.raw_os_error().unwrap() as u32;
                Ok(-1)
            }
        }
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
            if let Some(old_flag) = this.machine.file_handler.flags.get_mut(&fd) {
                if flag ^ *old_flag == fd_cloexec {
                    *old_flag = flag;
                } else {
                    throw_unsup_format!("Unsupported arg {:#x} for `F_SETFD`", flag);
                }
            }
            Ok(0)
        } else if cmd == this.eval_libc_i32("F_GETFD")? {
            if let Some(flag) = this.machine.file_handler.flags.get(&fd) {
                Ok(*flag)
            } else {
                this.machine.last_error = this.eval_libc_i32("EBADF")? as u32;
                Ok(-1)
            }
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

        if let Some(file) = this.machine.file_handler.files.remove(&fd) {
            match file.sync_all() {
                Ok(()) => Ok(0),
                Err(e) => {
                    this.machine.last_error = e.raw_os_error().unwrap() as u32;
                    Ok(-1)
                }
            }
        } else {
            this.machine.last_error = this.eval_libc_i32("EBADF")? as u32;
            Ok(-1)
        }
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

        if let Some(file) = this.machine.file_handler.files.get_mut(&fd) {
            let mut bytes = vec![0; count as usize];
            match file.read(&mut bytes) {
                Ok(read_bytes) => {
                    bytes.truncate(read_bytes);

                    this.memory_mut()
                        .get_mut(buf.alloc_id)?
                        .write_bytes(tcx, buf, &bytes)?;

                    Ok(read_bytes as i64)
                }
                Err(e) => {
                    this.machine.last_error = e.raw_os_error().unwrap() as u32;
                    Ok(-1)
                }
            }
        } else {
            this.machine.last_error = this.eval_libc_i32("EBADF")? as u32;
            Ok(-1)
        }
    }
}
