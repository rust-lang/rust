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

        match link_name {
            // Environment related shims
            "GetEnvironmentVariableW" => {
                // args[0] : LPCWSTR lpName (32-bit ptr to a const string of 16-bit Unicode chars)
                // args[1] : LPWSTR lpBuffer (32-bit pointer to a string of 16-bit Unicode chars)
                // lpBuffer : ptr to buffer that receives contents of the env_var as a null-terminated string.
                // Return `# of chars` stored in the buffer pointed to by lpBuffer, excluding null-terminator.
                // Return 0 upon failure.

                // This is not the env var you are looking for.
                this.set_last_error(Scalar::from_u32(203))?; // ERROR_ENVVAR_NOT_FOUND
                this.write_null(dest)?;
            }

            "SetEnvironmentVariableW" => {
                // args[0] : LPCWSTR lpName (32-bit ptr to a const string of 16-bit Unicode chars)
                // args[1] : LPCWSTR lpValue (32-bit ptr to a const string of 16-bit Unicode chars)
                // Return nonzero if success, else return 0.
                throw_unsup_format!("can't set environment variable on Windows");
            }

            // File related shims
            "WriteFile" => {
                let handle = this.read_scalar(args[0])?.to_machine_isize(this)?;
                let buf = this.read_scalar(args[1])?.not_undef()?;
                let n = this.read_scalar(args[2])?.to_u32()?;
                let written_place = this.deref_operand(args[3])?;
                // Spec says to always write `0` first.
                this.write_null(written_place.into())?;
                let written = if handle == -11 || handle == -12 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = this.memory.read_bytes(buf, Size::from_bytes(u64::from(n)))?;
                    let res = if handle == -11 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    res.ok().map(|n| n as u32)
                } else {
                    eprintln!("Miri: Ignored output to handle {}", handle);
                    // Pretend it all went well.
                    Some(n)
                };
                // If there was no error, write back how much was written.
                if let Some(n) = written {
                    this.write_scalar(Scalar::from_u32(n), written_place.into())?;
                }
                // Return whether this was a success.
                this.write_scalar(
                    Scalar::from_int(if written.is_some() { 1 } else { 0 }, dest.layout.size),
                    dest,
                )?;
            }
            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        }

        Ok(())
    }
}

