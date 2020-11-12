//! Unwind info generation (`.eh_frame`)

use crate::prelude::*;

use cranelift_codegen::isa::{unwind::UnwindInfo, TargetIsa};

use gimli::write::{Address, CieId, EhFrame, FrameTable, Section};

use crate::backend::WriteDebugInfo;

pub(crate) struct UnwindContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    frame_table: FrameTable,
    cie_id: Option<CieId>,
}

impl<'tcx> UnwindContext<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, isa: &dyn TargetIsa) -> Self {
        let mut frame_table = FrameTable::default();

        let cie_id = if let Some(mut cie) = isa.create_systemv_cie() {
            if isa.flags().is_pic() {
                cie.fde_address_encoding =
                    gimli::DwEhPe(gimli::DW_EH_PE_pcrel.0 | gimli::DW_EH_PE_sdata4.0);
            }
            Some(frame_table.add_cie(cie))
        } else {
            None
        };

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
                self.frame_table.add_fde(
                    self.cie_id.unwrap(),
                    unwind_info.to_fde(Address::Symbol {
                        symbol: func_id.as_u32() as usize,
                        addend: 0,
                    }),
                );
            }
            UnwindInfo::WindowsX64(_) => {
                // FIXME implement this
            }
            unwind_info => unimplemented!("{:?}", unwind_info),
        }
    }

    pub(crate) fn emit<P: WriteDebugInfo>(self, product: &mut P) {
        let mut eh_frame = EhFrame::from(super::emit::WriterRelocate::new(super::target_endian(
            self.tcx,
        )));
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

    #[cfg(feature = "jit")]
    pub(crate) unsafe fn register_jit(
        self,
        jit_product: &cranelift_simplejit::SimpleJITProduct,
    ) -> Option<UnwindRegistry> {
        let mut eh_frame = EhFrame::from(super::emit::WriterRelocate::new(super::target_endian(
            self.tcx,
        )));
        self.frame_table.write_eh_frame(&mut eh_frame).unwrap();

        if eh_frame.0.writer.slice().is_empty() {
            return None;
        }

        let mut eh_frame = eh_frame.0.relocate_for_jit(jit_product);

        // GCC expects a terminating "empty" length, so write a 0 length at the end of the table.
        eh_frame.extend(&[0, 0, 0, 0]);

        let mut registrations = Vec::new();

        // =======================================================================
        // Everything after this line up to the end of the file is loosly based on
        // https://github.com/bytecodealliance/wasmtime/blob/4471a82b0c540ff48960eca6757ccce5b1b5c3e4/crates/jit/src/unwind/systemv.rs
        #[cfg(target_os = "macos")]
        {
            // On macOS, `__register_frame` takes a pointer to a single FDE
            let start = eh_frame.as_ptr();
            let end = start.add(eh_frame.len());
            let mut current = start;

            // Walk all of the entries in the frame table and register them
            while current < end {
                let len = std::ptr::read::<u32>(current as *const u32) as usize;

                // Skip over the CIE
                if current != start {
                    __register_frame(current);
                    registrations.push(current as usize);
                }

                // Move to the next table entry (+4 because the length itself is not inclusive)
                current = current.add(len + 4);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // On other platforms, `__register_frame` will walk the FDEs until an entry of length 0
            let ptr = eh_frame.as_ptr();
            __register_frame(ptr);
            registrations.push(ptr as usize);
        }

        Some(UnwindRegistry {
            _frame_table: eh_frame,
            registrations,
        })
    }
}

/// Represents a registry of function unwind information for System V ABI.
pub(crate) struct UnwindRegistry {
    _frame_table: Vec<u8>,
    registrations: Vec<usize>,
}

extern "C" {
    // libunwind import
    fn __register_frame(fde: *const u8);
    fn __deregister_frame(fde: *const u8);
}

impl Drop for UnwindRegistry {
    fn drop(&mut self) {
        unsafe {
            // libgcc stores the frame entries as a linked list in decreasing sort order
            // based on the PC value of the registered entry.
            //
            // As we store the registrations in increasing order, it would be O(N^2) to
            // deregister in that order.
            //
            // To ensure that we just pop off the first element in the list upon every
            // deregistration, walk our list of registrations backwards.
            for fde in self.registrations.iter().rev() {
                __deregister_frame(*fde as *const _);
            }
        }
    }
}
