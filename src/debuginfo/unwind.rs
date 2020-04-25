use crate::prelude::*;

use cranelift_codegen::isa::unwind::UnwindInfo;

use gimli::write::Address;

impl<'a, 'tcx> FunctionDebugContext<'a, 'tcx> {
    pub(super) fn create_unwind_info(
        &mut self,
        context: &Context,
        isa: &dyn cranelift_codegen::isa::TargetIsa,
    ) {
        let unwind_info = if let Some(unwind_info) = context.create_unwind_info(isa).unwrap() {
            unwind_info
        } else {
            return;
        };

        match unwind_info {
            UnwindInfo::SystemV(unwind_info) => {
                self.debug_context.frame_table.add_fde(self.debug_context.cie, unwind_info.to_fde(Address::Symbol {
                    symbol: self.symbol,
                    addend: 0,
                }));
            },
            UnwindInfo::WindowsX64(_) => {
                // FIXME implement this
            }
        }
    }
}
