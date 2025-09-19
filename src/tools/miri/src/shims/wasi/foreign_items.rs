use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::alloc::EvalContextExt as _;
use crate::*;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Allocation
            "posix_memalign" => {
                let [memptr, align, size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.posix_memalign(memptr, align, size)?;
                this.write_scalar(result, dest)?;
            }
            "aligned_alloc" => {
                let [align, size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let res = this.aligned_alloc(align, size)?;
                this.write_pointer(res, dest)?;
            }

            // Standard input/output
            // FIXME: These shims are hacks that just get basic stdout/stderr working. We can't
            // constrain them to "std" since std itself uses the wasi crate for this.
            "get-stdout" => {
                let [] =
                    this.check_shim_sig(shim_sig!(extern "C" fn() -> i32), link_name, abi, args)?;
                this.write_scalar(Scalar::from_i32(1), dest)?; // POSIX FD number for stdout
            }
            "get-stderr" => {
                let [] =
                    this.check_shim_sig(shim_sig!(extern "C" fn() -> i32), link_name, abi, args)?;
                this.write_scalar(Scalar::from_i32(2), dest)?; // POSIX FD number for stderr
            }
            "[resource-drop]output-stream" => {
                let [handle] =
                    this.check_shim_sig(shim_sig!(extern "C" fn(i32) -> ()), link_name, abi, args)?;
                let handle = this.read_scalar(handle)?.to_i32()?;

                if !(handle == 1 || handle == 2) {
                    throw_unsup_format!("wasm output-stream: unsupported handle");
                }
                // We don't actually close these FDs, so this is a NOP.
            }
            "[method]output-stream.blocking-write-and-flush" => {
                let [handle, buf, len, ret_area] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *mut _, usize, *mut _) -> ()),
                    link_name,
                    abi,
                    args,
                )?;
                let handle = this.read_scalar(handle)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let len = this.read_target_usize(len)?;
                let ret_area = this.read_pointer(ret_area)?;

                if len > 4096 {
                    throw_unsup_format!(
                        "wasm output-stream.blocking-write-and-flush: buffer too big"
                    );
                }
                let len = usize::try_from(len).unwrap();
                let Some(fd) = this.machine.fds.get(handle) else {
                    throw_unsup_format!(
                        "wasm output-stream.blocking-write-and-flush: unsupported handle"
                    );
                };
                fd.write(
                    this.machine.communicate(),
                    buf,
                    len,
                    this,
                    callback!(
                        @capture<'tcx> {
                            len: usize,
                            ret_area: Pointer,
                        }
                        |this, result: Result<usize, IoError>| {
                            if !matches!(result, Ok(l) if l == len) {
                                throw_unsup_format!("wasm output-stream.blocking-write-and-flush: returning errors is not supported");
                            }
                            // 0 in the first byte of the ret_area indicates success.
                            let ret = this.ptr_to_mplace(ret_area, this.machine.layouts.u8);
                            this.write_null(&ret)?;
                            interp_ok(())
                    }),
                )?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
