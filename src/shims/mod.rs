pub mod foreign_items;
pub mod intrinsics;
pub mod tls;
pub mod dlsym;
pub mod env;
pub mod io;

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

            let n = {
                let ptr = this.force_ptr(this.read_scalar(args[0])?.not_undef()?)?;
                let align = this.force_bits(
                    this.read_scalar(args[1])?.not_undef()?,
                    this.pointer_size()
                )? as usize;

                let stride = this.memory().get(ptr.alloc_id)?.align.bytes() as usize;
                // if the allocation alignment is at least the required alignment, we use the
                // libcore implementation
                if stride >= align {
                    ((stride + ptr.offset.bytes() as usize) as *const ())
                        .align_offset(align) as u128
                } else {
                    u128::max_value()
                }
            };

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
}
