//! Unwind info generation (`.eh_frame`)

use cranelift_codegen::ir::Endianness;
use cranelift_codegen::isa::unwind::UnwindInfo;
use cranelift_object::ObjectProduct;
use gimli::RunTimeEndian;
use gimli::write::{CieId, EhFrame, FrameTable, Section};

use super::emit::address_for_func;
use super::object::WriteDebugInfo;
use crate::prelude::*;

pub(crate) struct UnwindContext {
    endian: RunTimeEndian,
    frame_table: FrameTable,
    cie_id: Option<CieId>,
}

impl UnwindContext {
    pub(crate) fn new(module: &mut dyn Module, pic_eh_frame: bool) -> Self {
        let endian = match module.isa().endianness() {
            Endianness::Little => RunTimeEndian::Little,
            Endianness::Big => RunTimeEndian::Big,
        };
        let mut frame_table = FrameTable::default();

        let cie_id = if let Some(mut cie) = module.isa().create_systemv_cie() {
            if pic_eh_frame {
                cie.fde_address_encoding =
                    gimli::DwEhPe(gimli::DW_EH_PE_pcrel.0 | gimli::DW_EH_PE_sdata4.0);
            }
            Some(frame_table.add_cie(cie))
        } else {
            None
        };

        UnwindContext { endian, frame_table, cie_id }
    }

    pub(crate) fn add_function(
        &mut self,
        module: &mut dyn Module,
        func_id: FuncId,
        context: &Context,
    ) {
        if let target_lexicon::OperatingSystem::MacOSX { .. } =
            module.isa().triple().operating_system
        {
            // The object crate doesn't currently support DW_GNU_EH_PE_absptr, which macOS
            // requires for unwinding tables. In addition on arm64 it currently doesn't
            // support 32bit relocations as we currently use for the unwinding table.
            // See gimli-rs/object#415 and rust-lang/rustc_codegen_cranelift#1371
            return;
        }

        let unwind_info = if let Some(unwind_info) =
            context.compiled_code().unwrap().create_unwind_info(module.isa()).unwrap()
        {
            unwind_info
        } else {
            return;
        };

        match unwind_info {
            UnwindInfo::SystemV(unwind_info) => {
                self.frame_table
                    .add_fde(self.cie_id.unwrap(), unwind_info.to_fde(address_for_func(func_id)));
            }
            UnwindInfo::WindowsX64(_) | UnwindInfo::WindowsArm64(_) => {
                // Windows does not have debug info for its unwind info.
            }
            unwind_info => unimplemented!("{:?}", unwind_info),
        }
    }

    pub(crate) fn emit(self, product: &mut ObjectProduct) {
        let mut eh_frame = EhFrame::from(super::emit::WriterRelocate::new(self.endian));
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

    #[cfg(all(feature = "jit", windows))]
    pub(crate) unsafe fn register_jit(self, _jit_module: &cranelift_jit::JITModule) {}

    #[cfg(all(feature = "jit", not(windows)))]
    pub(crate) unsafe fn register_jit(self, jit_module: &cranelift_jit::JITModule) {
        use std::mem::ManuallyDrop;

        let mut eh_frame = EhFrame::from(super::emit::WriterRelocate::new(self.endian));
        self.frame_table.write_eh_frame(&mut eh_frame).unwrap();

        if eh_frame.0.writer.slice().is_empty() {
            return;
        }

        let mut eh_frame = eh_frame.0.relocate_for_jit(jit_module);

        // GCC expects a terminating "empty" length, so write a 0 length at the end of the table.
        eh_frame.extend(&[0, 0, 0, 0]);

        // FIXME support unregistering unwind tables once cranelift-jit supports deallocating
        // individual functions
        let eh_frame = ManuallyDrop::new(eh_frame);

        // =======================================================================
        // Everything after this line up to the end of the file is loosely based on
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
                }

                // Move to the next table entry (+4 because the length itself is not inclusive)
                current = current.add(len + 4);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // On other platforms, `__register_frame` will walk the FDEs until an entry of length 0
            __register_frame(eh_frame.as_ptr());
        }
    }
}

extern "C" {
    // libunwind import
    fn __register_frame(fde: *const u8);
}
