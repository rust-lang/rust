mod linux;
mod macos;

use crate::*;
use rustc::ty::layout::{Align, Size};

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

            "posix_memalign" => {
                let ret = this.deref_operand(args[0])?;
                let align = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let size = this.read_scalar(args[2])?.to_machine_usize(this)?;
                // Align must be power of 2, and also at least ptr-sized (POSIX rules).
                if !align.is_power_of_two() {
                    throw_unsup!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                if align < this.pointer_size().bytes() {
                    throw_ub_format!(
                        "posix_memalign: alignment must be at least the size of a pointer, but is {}",
                        align,
                    );
                }

                if size == 0 {
                    this.write_null(ret.into())?;
                } else {
                    let ptr = this.memory.allocate(
                        Size::from_bytes(size),
                        Align::from_bytes(align).unwrap(),
                        MiriMemoryKind::C.into(),
                    );
                    this.write_scalar(ptr, ret.into())?;
                }
                this.write_null(dest)?;
            }

            "dlsym" => {
                let _handle = this.read_scalar(args[0])?;
                let symbol = this.read_scalar(args[1])?.not_undef()?;
                let symbol_name = this.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                if let Some(dlsym) = Dlsym::from_str(symbol_name)? {
                    let ptr = this.memory.create_fn_alloc(FnVal::Other(dlsym));
                    this.write_scalar(Scalar::from(ptr), dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            "memrchr" => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let val = this.read_scalar(args[1])?.to_i32()? as u8;
                let num = this.read_scalar(args[2])?.to_machine_usize(this)?;
                if let Some(idx) = this
                    .memory
                    .read_bytes(ptr, Size::from_bytes(num))?
                    .iter()
                    .rev()
                    .position(|&c| c == val)
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(num - idx as u64 - 1), this)?;
                    this.write_scalar(new_ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
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

// Shims the posix 'getrandom()' syscall.
fn getrandom<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    let ptr = this.read_scalar(args[0])?.not_undef()?;
    let len = this.read_scalar(args[1])?.to_machine_usize(this)?;

    // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
    // neither of which have any effect on our current PRNG.
    let _flags = this.read_scalar(args[2])?.to_i32()?;

    this.gen_random(ptr, len as usize)?;
    this.write_scalar(Scalar::from_uint(len, dest.layout.size), dest)?;
    Ok(())
}
