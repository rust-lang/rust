// .debug_gdb_scripts binary section.

use crate::llvm;

use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::value::Value;
use rustc_codegen_ssa::traits::*;
use rustc_middle::bug;
use rustc_session::config::DebugInfo;

use rustc_span::symbol::sym;

/// Inserts a side-effect free instruction sequence that makes sure that the
/// .debug_gdb_scripts global is referenced, so it isn't removed by the linker.
pub fn insert_reference_to_gdb_debug_scripts_section_global(bx: &mut Builder<'_, '_, '_>) {
    if needs_gdb_debug_scripts_section(bx) {
        let gdb_debug_scripts_section = get_or_insert_gdb_debug_scripts_section_global(bx);
        // Load just the first byte as that's all that's necessary to force
        // LLVM to keep around the reference to the global.
        let indices = [bx.const_i32(0), bx.const_i32(0)];
        let element = bx.inbounds_gep(gdb_debug_scripts_section, &indices);
        let volative_load_instruction = bx.volatile_load(element);
        unsafe {
            llvm::LLVMSetAlignment(volative_load_instruction, 1);
        }
    }
}

/// Allocates the global variable responsible for the .debug_gdb_scripts binary
/// section.
pub fn get_or_insert_gdb_debug_scripts_section_global(cx: &CodegenCx<'ll, '_>) -> &'ll Value {
    let c_section_var_name = "__rustc_debug_gdb_scripts_section__\0";
    let section_var_name = &c_section_var_name[..c_section_var_name.len() - 1];

    let section_var =
        unsafe { llvm::LLVMGetNamedGlobal(cx.llmod, c_section_var_name.as_ptr().cast()) };

    section_var.unwrap_or_else(|| {
        let section_name = b".debug_gdb_scripts\0";
        let section_contents = b"\x01gdb_load_rust_pretty_printers.py\0";

        unsafe {
            let llvm_type = cx.type_array(cx.type_i8(), section_contents.len() as u64);

            let section_var = cx
                .define_global(section_var_name, llvm_type)
                .unwrap_or_else(|| bug!("symbol `{}` is already defined", section_var_name));
            llvm::LLVMSetSection(section_var, section_name.as_ptr().cast());
            llvm::LLVMSetInitializer(section_var, cx.const_bytes(section_contents));
            llvm::LLVMSetGlobalConstant(section_var, llvm::True);
            llvm::LLVMSetUnnamedAddress(section_var, llvm::UnnamedAddr::Global);
            llvm::LLVMRustSetLinkage(section_var, llvm::Linkage::LinkOnceODRLinkage);
            // This should make sure that the whole section is not larger than
            // the string it contains. Otherwise we get a warning from GDB.
            llvm::LLVMSetAlignment(section_var, 1);
            section_var
        }
    })
}

pub fn needs_gdb_debug_scripts_section(cx: &CodegenCx<'_, '_>) -> bool {
    let omit_gdb_pretty_printer_section = cx
        .tcx
        .sess
        .contains_name(&cx.tcx.hir().krate_attrs(), sym::omit_gdb_pretty_printer_section);

    !omit_gdb_pretty_printer_section
        && cx.sess().opts.debuginfo != DebugInfo::None
        && cx.sess().target.options.emit_debug_gdb_scripts
}
