use crate::prelude::*;

use cranelift_codegen::isa::{TargetIsa, unwind::UnwindInfo};

use gimli::write::{Address, CieId, EhFrame, FrameTable, Section};

use crate::backend::WriteDebugInfo;

pub(crate) struct UnwindContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    frame_table: FrameTable,
    cie_id: CieId,
}

impl<'tcx> UnwindContext<'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        module: &mut Module<impl Backend>,
    ) -> Self {
        let mut frame_table = FrameTable::default();
        let cie = module.isa().create_systemv_cie().expect("SystemV unwind info CIE");

        let cie_id = frame_table.add_cie(cie);

        UnwindContext {
            tcx,
            frame_table,
            cie_id,
        }
    }

    pub(crate) fn add_function(&mut self, func_id: FuncId, context: &Context, isa: &dyn TargetIsa) {
        let unwind_info = if let Some(unwind_info) = context.create_unwind_info(isa).unwrap() {
            unwind_info
        } else {
            return;
        };

        match unwind_info {
            UnwindInfo::SystemV(unwind_info) => {
                self.frame_table.add_fde(self.cie_id, unwind_info.to_fde(Address::Symbol {
                    symbol: func_id.as_u32() as usize,
                    addend: 0,
                }));
            },
            UnwindInfo::WindowsX64(_) => {
                // FIXME implement this
            }
        }
    }

    pub(crate) fn emit<P: WriteDebugInfo>(self, product: &mut P) {
        let mut eh_frame = EhFrame::from(super::emit::WriterRelocate::new(super::target_endian(self.tcx)));
        self.frame_table.write_eh_frame(&mut eh_frame).unwrap();

        if !eh_frame.0.writer.slice().is_empty() {
            let id = eh_frame.id();
            let section_id = product.add_debug_section(id, eh_frame.0.writer.into_vec());
            let mut section_map = FxHashMap::default();
            section_map.insert(id, section_id);

            for reloc in &eh_frame.0.relocs {
                product.add_debug_reloc(&section_map, &section_id, reloc);
            }
        }
    }
}
