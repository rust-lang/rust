#![warn(clippy::arithmetic_side_effects)]

mod backtrace;
#[cfg(target_os = "linux")]
pub mod ffi_support;
pub mod foreign_items;
pub mod intrinsics;
pub mod unix;
pub mod windows;
mod x86;

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

        // For foreign items, try to see if we can emulate them.
        if this.tcx.is_foreign_item(instance.def_id()) {
            // An external function call that does not have a MIR body. We either find MIR elsewhere
            // or emulate its effect.
            // This will be Ok(None) if we're emulating the intrinsic entirely within Miri (no need
            // to run extra MIR), and Ok(Some(body)) if we found MIR to run for the
            // foreign function
            // Any needed call to `goto_block` will be performed by `emulate_foreign_item`.
            let args = this.copy_fn_args(args)?; // FIXME: Should `InPlace` arguments be reset to uninit?
            let link_name = this.item_link_name(instance.def_id());
            return this.emulate_foreign_item(link_name, abi, &args, dest, ret, unwind);
        }

        // Otherwise, load the MIR.
        Ok(Some((this.load_mir(instance.def, None)?, instance)))
    }
}
