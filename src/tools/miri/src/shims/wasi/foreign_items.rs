use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

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
        abi: Abi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Allocation
            "posix_memalign" => {
                let [memptr, align, size] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.posix_memalign(memptr, align, size)?;
                this.write_scalar(result, dest)?;
            }
            "aligned_alloc" => {
                let [align, size] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let res = this.aligned_alloc(align, size)?;
                this.write_pointer(res, dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
