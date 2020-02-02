mod linux;
mod macos;

use crate::*;
use rustc::ty::layout::Size;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let tcx = &{ this.tcx.tcx };

        match link_name {
            // Environment related shims
            "getenv" => {
                let result = this.getenv(args[0])?;
                this.write_scalar(result, dest)?;
            }

            "unsetenv" => {
                let result = this.unsetenv(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "setenv" => {
                let result = this.setenv(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "getcwd" => {
                let result = this.getcwd(args[0], args[1])?;
                this.write_scalar(result, dest)?;
            }

            "chdir" => {
                let result = this.chdir(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // File related shims
            "fcntl" => {
                let result = this.fcntl(args[0], args[1], args.get(2).cloned())?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "read" => {
                let result = this.read(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "write" => {
                let fd = this.read_scalar(args[0])?.to_i32()?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_machine_usize(tcx)?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = this.memory.read_bytes(buf, Size::from_bytes(n))?;
                    // We need to flush to make sure this actually appears on the screen
                    let res = if fd == 1 {
                        // Stdout is buffered, flush to make sure it appears on the screen.
                        // This is the write() syscall of the interpreted program, we want it
                        // to correspond to a write() syscall on the host -- there is no good
                        // in adding extra buffering here.
                        let res = io::stdout().write(buf_cont);
                        io::stdout().flush().unwrap();
                        res
                    } else {
                        // No need to flush, stderr is not buffered.
                        io::stderr().write(buf_cont)
                    };
                    match res {
                        Ok(n) => n as i64,
                        Err(_) => -1,
                    }
                } else {
                    this.write(args[0], args[1], args[2])?
                };
                // Now, `result` is the value we return back to the program.
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "unlink" => {
                let result = this.unlink(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "symlink" => {
                let result = this.symlink(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            _ => {
                match this.tcx.sess.target.target.target_os.to_lowercase().as_str() {
                    "linux" => linux::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest)?,
                    "macos" => macos::EvalContextExt::emulate_foreign_item_by_name(this, link_name, args, dest)?,
                    _ => unreachable!(),
                }
            }
        };

        Ok(())
    }
}
