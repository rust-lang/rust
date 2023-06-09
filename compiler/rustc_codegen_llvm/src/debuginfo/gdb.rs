// .debug_gdb_scripts binary section.

use crate::llvm;

use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::value::Value;
use rustc_ast::attr;
use rustc_codegen_ssa::base::collect_debugger_visualizers_transitive;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::{bug, middle::debugger_visualizer::DebuggerVisualizerType};
use rustc_session::config::{CrateType, DebugInfo};
use rustc_span::symbol::sym;

/// Inserts a side-effect free instruction sequence that makes sure that the
/// .debug_gdb_scripts global is referenced, so it isn't removed by the linker.
pub fn insert_reference_to_gdb_debug_scripts_section_global(bx: &mut Builder<'_, '_, '_>) {
    if needs_gdb_debug_scripts_section(bx) {
        let gdb_debug_scripts_section =
            bx.const_bitcast(get_or_insert_gdb_debug_scripts_section_global(bx), bx.type_i8p());
        // Load just the first byte as that's all that's necessary to force
        // LLVM to keep around the reference to the global.
        let volatile_load_instruction = bx.volatile_load(bx.type_i8(), gdb_debug_scripts_section);
        unsafe {
            llvm::LLVMSetAlignment(volatile_load_instruction, 1);
        }
    }
}

/// Allocates the global variable responsible for the .debug_gdb_scripts binary
/// section.
pub fn get_or_insert_gdb_debug_scripts_section_global<'ll>(cx: &CodegenCx<'ll, '_>) -> &'ll Value {
    let c_section_var_name = "__rustc_debug_gdb_scripts_section__\0";
    let section_var_name = &c_section_var_name[..c_section_var_name.len() - 1];

    let section_var =
        unsafe { llvm::LLVMGetNamedGlobal(cx.llmod, c_section_var_name.as_ptr().cast()) };

    section_var.unwrap_or_else(|| {
        let mut section_contents = Vec::new();

        // Add the pretty printers for the standard library first.
        section_contents.extend_from_slice(b"\x01gdb_load_rust_pretty_printers.py\0");

        // Next, add the pretty printers that were specified via the `#[debugger_visualizer]` attribute.
        let visualizers = collect_debugger_visualizers_transitive(
            cx.tcx,
            DebuggerVisualizerType::GdbPrettyPrinter,
        );
        let crate_name = cx.tcx.crate_name(LOCAL_CRATE);
        for (index, visualizer) in visualizers.iter().enumerate() {
            // The initial byte `4` instructs GDB that the following pretty printer
            // is defined inline as opposed to in a standalone file.
            section_contents.extend_from_slice(b"\x04");
            let vis_name = format!("pretty-printer-{}-{}\n", crate_name, index);
            section_contents.extend_from_slice(vis_name.as_bytes());
            section_contents.extend_from_slice(&visualizer.src);

            // The final byte `0` tells GDB that the pretty printer has been
            // fully defined and can continue searching for additional
            // pretty printers.
            section_contents.extend_from_slice(b"\0");
        }

        unsafe {
            let section_contents = section_contents.as_slice();
            let llvm_type = cx.type_array(cx.type_i8(), section_contents.len() as u64);

            let section_var = cx
                .define_global(section_var_name, llvm_type)
                .unwrap_or_else(|| bug!("symbol `{}` is already defined", section_var_name));
            llvm::LLVMSetSection(section_var, c".debug_gdb_scripts".as_ptr().cast());
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
    let omit_gdb_pretty_printer_section =
        attr::contains_name(cx.tcx.hir().krate_attrs(), sym::omit_gdb_pretty_printer_section);

    // To ensure the section `__rustc_debug_gdb_scripts_section__` will not create
    // ODR violations at link time, this section will not be emitted for rlibs since
    // each rlib could produce a different set of visualizers that would be embedded
    // in the `.debug_gdb_scripts` section. For that reason, we make sure that the
    // section is only emitted for leaf crates.
    let embed_visualizers = cx.sess().crate_types().iter().any(|&crate_type| match crate_type {
        CrateType::Executable | CrateType::Dylib | CrateType::Cdylib | CrateType::Staticlib => {
            // These are crate types for which we will embed pretty printers since they
            // are treated as leaf crates.
            true
        }
        CrateType::ProcMacro => {
            // We could embed pretty printers for proc macro crates too but it does not
            // seem like a good default, since this is a rare use case and we don't
            // want to slow down the common case.
            false
        }
        CrateType::Rlib => {
            // As per the above description, embedding pretty printers for rlibs could
            // lead to ODR violations so we skip this crate type as well.
            false
        }
    });

    !omit_gdb_pretty_printer_section
        && cx.sess().opts.debuginfo != DebugInfo::None
        && cx.sess().target.emit_debug_gdb_scripts
        && embed_visualizers
}
