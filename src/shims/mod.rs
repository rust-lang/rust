pub mod dlsym;
pub mod env;
pub mod foreign_items;
pub mod fs;
pub mod intrinsics;
pub mod panic;
pub mod time;
pub mod tls;

use crate::*;
use rustc::{mir, ty};

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn find_mir_or_eval_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        ret: Option<(PlaceTy<'tcx, Tag>, mir::BasicBlock)>,
        unwind: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        let this = self.eval_context_mut();
        trace!("eval_fn_call: {:#?}, {:?}", instance, ret.map(|p| *p.0));

        // There are some more lang items we want to hook that CTFE does not hook (yet).
        if this.tcx.lang_items().align_offset_fn() == Some(instance.def.def_id()) {
            let (dest, ret) = ret.unwrap();
            let n = this
                .align_offset(args[0], args[1])?
                .unwrap_or_else(|| this.truncate(u128::max_value(), dest.layout));
            this.write_scalar(Scalar::from_uint(n, dest.layout.size), dest)?;
            this.go_to_block(ret);
            return Ok(None);
        }

        // Try to see if we can do something about foreign items.
        if this.tcx.is_foreign_item(instance.def_id()) {
            // An external function call that does not have a MIR body. We either find MIR elsewhere
            // or emulate its effect.
            // This will be Ok(None) if we're emulating the intrinsic entirely within Miri (no need
            // to run extra MIR), and Ok(Some(body)) if we found MIR to run for the
            // foreign function
            // Any needed call to `goto_block` will be performed by `emulate_foreign_item`.
            return this.emulate_foreign_item(instance.def_id(), args, ret, unwind);
        }

        // Otherwise, load the MIR.
        Ok(Some(&*this.load_mir(instance.def, None)?))
    }

    fn align_offset(
        &mut self,
        ptr_op: OpTy<'tcx, Tag>,
        align_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, Option<u128>> {
        let this = self.eval_context_mut();

        let req_align = this
            .force_bits(this.read_scalar(align_op)?.not_undef()?, this.pointer_size())?
            as usize;

        // FIXME: This should actually panic in the interpreted program
        if !req_align.is_power_of_two() {
            throw_unsup_format!("Required alignment should always be a power of two")
        }

        let ptr_scalar = this.read_scalar(ptr_op)?.not_undef()?;

        if let Ok(ptr) = this.force_ptr(ptr_scalar) {
            let cur_align =
                this.memory.get_size_and_align(ptr.alloc_id, AllocCheck::MaybeDead)?.1.bytes()
                    as usize;
            if cur_align >= req_align {
                // if the allocation alignment is at least the required alignment we use the
                // libcore implementation
                return Ok(Some(
                    (this.force_bits(ptr_scalar, this.pointer_size())? as *const i8)
                        .align_offset(req_align) as u128,
                ));
            }
        }
        // If the allocation alignment is smaller than then required alignment or the pointer was
        // actually an integer, we return `None`
        Ok(None)
    }
}
