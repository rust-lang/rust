use rustc_middle::mir;
use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use super::horizontal_bin_op;
use crate::*;
use shims::foreign_items::EmulateForeignItemResult;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub(super) trait EvalContextExt<'mir, 'tcx: 'mir>:
    crate::MiriInterpCxExt<'mir, 'tcx>
{
    fn emulate_x86_sse3_intrinsic(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateForeignItemResult> {
        let this = self.eval_context_mut();
        // Prefix should have already been checked.
        let unprefixed_name = link_name.as_str().strip_prefix("llvm.x86.sse3.").unwrap();

        match unprefixed_name {
            // Used to implement the _mm_h{add,sub}_p{s,d} functions.
            // Horizontally add/subtract adjacent floating point values
            // in `left` and `right`.
            "hadd.ps" | "hadd.pd" | "hsub.ps" | "hsub.pd" => {
                let [left, right] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;

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
                let [src_ptr] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let src_ptr = this.read_pointer(src_ptr)?;
                let dest = dest.force_mplace(this)?;

                this.mem_copy(src_ptr, dest.ptr(), dest.layout.size, /*nonoverlapping*/ true)?;
            }
            _ => return Ok(EmulateForeignItemResult::NotSupported),
        }
        Ok(EmulateForeignItemResult::NeedsJumping)
    }
}
