#![warn(clippy::arithmetic_side_effects)]

mod backtrace;
#[cfg(target_os = "linux")]
pub mod ffi_support;
pub mod foreign_items;
pub mod intrinsics;
pub mod unix;
pub mod windows;

pub mod dlsym;
pub mod env;
pub mod os_str;
pub mod panic;
pub mod time;
pub mod tls;

// End module management, begin local code

use log::trace;

use rustc_middle::{mir, ty};
use rustc_target::spec::abi::Abi;

use crate::*;
use helpers::check_arg_count;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn find_mir_or_eval_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        abi: Abi,
        args: &[FnArg<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<(&'mir mir::Body<'tcx>, ty::Instance<'tcx>)>> {
        let this = self.eval_context_mut();
        trace!("eval_fn_call: {:#?}, {:?}", instance, dest);

        // There are some more lang items we want to hook that CTFE does not hook (yet).
        if this.tcx.lang_items().align_offset_fn() == Some(instance.def.def_id()) {
            let args = this.copy_fn_args(args)?;
            let [ptr, align] = check_arg_count(&args)?;
            if this.align_offset(ptr, align, dest, ret, unwind)? {
                return Ok(None);
            }
        }

        // Try to see if we can do something about foreign items.
        if this.tcx.is_foreign_item(instance.def_id()) {
            // An external function call that does not have a MIR body. We either find MIR elsewhere
            // or emulate its effect.
            // This will be Ok(None) if we're emulating the intrinsic entirely within Miri (no need
            // to run extra MIR), and Ok(Some(body)) if we found MIR to run for the
            // foreign function
            // Any needed call to `goto_block` will be performed by `emulate_foreign_item`.
            let args = this.copy_fn_args(args)?; // FIXME: Should `InPlace` arguments be reset to uninit?
            return this.emulate_foreign_item(instance.def_id(), abi, &args, dest, ret, unwind);
        }

        // Otherwise, load the MIR.
        Ok(Some((this.load_mir(instance.def, None)?, instance)))
    }

    /// Returns `true` if the computation was performed, and `false` if we should just evaluate
    /// the actual MIR of `align_offset`.
    fn align_offset(
        &mut self,
        ptr_op: &OpTy<'tcx, Provenance>,
        align_op: &OpTy<'tcx, Provenance>,
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let ret = ret.unwrap();

        if this.machine.check_alignment != AlignmentCheck::Symbolic {
            // Just use actual implementation.
            return Ok(false);
        }

        let req_align = this.read_target_usize(align_op)?;

        // Stop if the alignment is not a power of two.
        if !req_align.is_power_of_two() {
            this.start_panic("align_offset: align is not a power-of-two", unwind)?;
            return Ok(true); // nothing left to do
        }

        let ptr = this.read_pointer(ptr_op)?;
        // If this carries no provenance, treat it like an integer.
        if ptr.provenance.is_none() {
            // Use actual implementation.
            return Ok(false);
        }

        if let Ok((alloc_id, _offset, _)) = this.ptr_try_get_alloc_id(ptr) {
            // Only do anything if we can identify the allocation this goes to.
            let (_size, cur_align, _kind) = this.get_alloc_info(alloc_id);
            if cur_align.bytes() >= req_align {
                // If the allocation alignment is at least the required alignment we use the
                // real implementation.
                return Ok(false);
            }
        }

        // Return error result (usize::MAX), and jump to caller.
        this.write_scalar(Scalar::from_target_usize(this.target_usize_max(), this), dest)?;
        this.go_to_block(ret);
        Ok(true)
    }
}
