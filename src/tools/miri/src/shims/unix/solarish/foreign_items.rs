use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Miscellaneous
            "___errno" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            "stack_getbounds" => {
                let [stack] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let stack = this.deref_pointer_as(stack, this.libc_ty_layout("stack_t"))?;

                this.write_int_fields_named(
                    &[
                        ("ss_sp", this.machine.stack_addr.into()),
                        ("ss_size", this.machine.stack_size.into()),
                        // field set to 0 means not in an alternate signal stack
                        // https://docs.oracle.com/cd/E86824_01/html/E54766/stack-getbounds-3c.html
                        ("ss_flags", 0),
                    ],
                    &stack,
                )?;

                this.write_null(dest)?;
            }

            _ => return Ok(EmulateItemResult::NotSupported),
        }
        Ok(EmulateItemResult::NeedsJumping)
    }
}
