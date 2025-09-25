use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr::NonNull;

use rustc_data_structures::small_c_str::SmallCStr;

use crate::errors::LlvmError;
use crate::llvm;

/// Responsible for safely creating and disposing llvm::TargetMachine via ffi functions.
/// Not cloneable as there is no clone function for llvm::TargetMachine.
#[repr(transparent)]
pub struct OwnedTargetMachine {
    tm_unique: NonNull<llvm::TargetMachine>,
    phantom: PhantomData<llvm::TargetMachine>,
}

impl OwnedTargetMachine {
    pub(crate) fn new(
        triple: &CStr,
        cpu: &CStr,
        features: &CStr,
        abi: &CStr,
        model: llvm::CodeModel,
        reloc: llvm::RelocModel,
        level: llvm::CodeGenOptLevel,
        float_abi: llvm::FloatAbi,
        function_sections: bool,
        data_sections: bool,
        unique_section_names: bool,
        trap_unreachable: bool,
        singlethread: bool,
        verbose_asm: bool,
        emit_stack_size_section: bool,
        relax_elf_relocations: bool,
        use_init_array: bool,
        split_dwarf_file: &CStr,
        output_obj_file: &CStr,
        debug_info_compression: &CStr,
        use_emulated_tls: bool,
        argv0: &str,
        command_line_args: &str,
        use_wasm_eh: bool,
    ) -> Result<Self, LlvmError<'static>> {
        // SAFETY: llvm::LLVMRustCreateTargetMachine copies pointed to data
        let tm_ptr = unsafe {
            llvm::LLVMRustCreateTargetMachine(
                triple.as_ptr(),
                cpu.as_ptr(),
                features.as_ptr(),
                abi.as_ptr(),
                model,
                reloc,
                level,
                float_abi,
                function_sections,
                data_sections,
                unique_section_names,
                trap_unreachable,
                singlethread,
                verbose_asm,
                emit_stack_size_section,
                relax_elf_relocations,
                use_init_array,
                split_dwarf_file.as_ptr(),
                output_obj_file.as_ptr(),
                debug_info_compression.as_ptr(),
                use_emulated_tls,
                argv0.as_ptr(),
                argv0.len(),
                command_line_args.as_ptr(),
                command_line_args.len(),
                use_wasm_eh,
            )
        };

        NonNull::new(tm_ptr)
            .map(|tm_unique| Self { tm_unique, phantom: PhantomData })
            .ok_or_else(|| LlvmError::CreateTargetMachine { triple: SmallCStr::from(triple) })
    }

    /// Returns inner `llvm::TargetMachine` type.
    ///
    /// This could be a `Deref` implementation, but `llvm::TargetMachine` is an extern type and
    /// `Deref::Target: ?Sized`.
    pub fn raw(&self) -> &llvm::TargetMachine {
        // SAFETY: constructing ensures we have a valid pointer created by
        // llvm::LLVMRustCreateTargetMachine.
        unsafe { self.tm_unique.as_ref() }
    }
}

impl Drop for OwnedTargetMachine {
    fn drop(&mut self) {
        // SAFETY: constructing ensures we have a valid pointer created by
        // llvm::LLVMRustCreateTargetMachine OwnedTargetMachine is not copyable so there is no
        // double free or use after free.
        unsafe { llvm::LLVMDisposeTargetMachine(self.tm_unique) };
    }
}
