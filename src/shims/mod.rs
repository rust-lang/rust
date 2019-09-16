pub mod foreign_items;
pub mod intrinsics;
pub mod tls;
pub mod dlsym;
pub mod env;

use rustc::{ty, mir};

use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn find_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir mir::Body<'tcx>>> {
        let this = self.eval_context_mut();
        trace!("eval_fn_call: {:#?}, {:?}", instance, dest.map(|place| *place));

        // First, run the common hooks also supported by CTFE.
        if this.hook_fn(instance, args, dest)? {
            this.goto_block(ret)?;
            return Ok(None);
        }
        // There are some more lang items we want to hook that CTFE does not hook (yet).
        if this.tcx.lang_items().align_offset_fn() == Some(instance.def.def_id()) {
            let n = this.align_offset(args[0], args[1])?;
            let dest = dest.unwrap();
            let n = this.truncate(n, dest.layout);
            this.write_scalar(Scalar::from_uint(n, dest.layout.size), dest)?;
            this.goto_block(ret)?;
            return Ok(None);
        }

        // Try to see if we can do something about foreign items.
        if this.tcx.is_foreign_item(instance.def_id()) {
            // An external function that we cannot find MIR for, but we can still run enough
            // of them to make miri viable.
            this.emulate_foreign_item(instance.def_id(), args, dest, ret)?;
            // `goto_block` already handled.
            return Ok(None);
        }

        // Otherwise, load the MIR.
        Ok(Some(this.load_mir(instance.def, None)?))
    }

    fn align_offset(
        &mut self,
        ptr_op: OpTy<'tcx, Tag>,
        align_op: OpTy<'tcx, Tag>
    ) -> InterpResult<'tcx, u128> {
        let this = self.eval_context_mut();

        let req_align = this.force_bits(
            this.read_scalar(align_op)?.not_undef()?,
            this.pointer_size()
        )? as usize;

        let ptr_scalar = this.read_scalar(ptr_op)?.not_undef()?;

        if let Scalar::Ptr(ptr) = ptr_scalar {
            let cur_align = this.memory().get(ptr.alloc_id)?.align.bytes() as usize;
            if cur_align < req_align {
                return Ok(u128::max_value());
            }
        }

        // if the allocation alignment is at least the required alignment or if the pointer is an
        // integer, we use the libcore implementation
        Ok(
            (this.force_bits(ptr_scalar, this.pointer_size())? as *const i8)
            .align_offset(req_align) as u128
        )
    }
}
