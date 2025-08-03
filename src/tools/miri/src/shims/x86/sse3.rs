use rustc_abi::CanonAbi;
use rustc_middle::mir;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use super::horizontal_bin_op;
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_x86_sse3_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        this.expect_target_feature_for_intrinsic(link_name, "sse3")?;
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse3.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm_h{add,sub}_p{s,d} functions.
            // Horizontally add/subtract adjacent floating point values
            // in `left` and `right`.
            "hadd.ps" | "hadd.pd" | "hsub.ps" | "hsub.pd" => {
                let [left, right] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;

                let which = match unprefixed_name {
                    "hadd.ps" | "hadd.pd" => mir::BinOp::Add,
                    "hsub.ps" | "hsub.pd" => mir::BinOp::Sub,
                    _ => unreachable!(),
                };

                horizontal_bin_op(this, which, /*saturating*/ false, left, right, dest)?;
            }
            // Used to implement the _mm_lddqu_si128 function.
            // Reads a 128-bit vector from an unaligned pointer. This intrinsic
            // is expected to perform better than a regular unaligned read when
            // the data crosses a cache line, but for Miri this is just a regular
            // unaligned read.
            "ldu.dq" => {
                let [src_ptr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let src_ptr = this.read_pointer(src_ptr)?;
                let dest = dest.force_mplace(this)?;

                this.mem_copy(src_ptr, dest.ptr(), dest.layout.size, /*nonoverlapping*/ true)?;
            }
            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
