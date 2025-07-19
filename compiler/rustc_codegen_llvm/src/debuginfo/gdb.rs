// .debug_gdb_scripts binary section.

use std::ffi::CString;

use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_codegen_ssa::base::collect_debugger_visualizers_transitive;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerType;
use rustc_session::config::{CrateType, DebugInfo};

use crate::common::CodegenCx;
use crate::llvm;
use crate::value::Value;

/// Allocates the global variable responsible for the .debug_gdb_scripts binary
/// section.
pub(crate) fn get_or_insert_gdb_debug_scripts_section_global<'ll>(
    cx: &mut CodegenCx<'ll, '_>,
) -> &'ll Value {
    let c_section_var_name = CString::new(format!(
        "__rustc_debug_gdb_scripts_section_{}_{:08x}",
        cx.tcx.crate_name(LOCAL_CRATE),
        cx.tcx.stable_crate_id(LOCAL_CRATE),
    ))
    .unwrap();
    let section_var_name = c_section_var_name.to_str().unwrap();

    let section_var = unsafe { llvm::LLVMGetNamedGlobal(cx.llmod, c_section_var_name.as_ptr()) };

    section_var.unwrap_or_else(|| {
        let mut section_contents = Vec::new();

        // Add the pretty printers for the standard library first.
        section_contents.extend_from_slice(b"\x01gdb_load_rust_pretty_printers.py\0");

        // Next, add the pretty printers that were specified via the `#[debugger_visualizer]`
        // attribute.
        let visualizers = collect_debugger_visualizers_transitive(
            cx.tcx,
            DebuggerVisualizerType::GdbPrettyPrinter,
        );
        let crate_name = cx.tcx.crate_name(LOCAL_CRATE);
        for (index, visualizer) in visualizers.iter().enumerate() {
            // The initial byte `4` instructs GDB that the following pretty printer
            // is defined inline as opposed to in a standalone file.
            section_contents.extend_from_slice(b"\x04");
            let vis_name = format!("pretty-printer-{crate_name}-{index}\n");
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
            llvm::set_section(section_var, c".debug_gdb_scripts");
            llvm::set_initializer(section_var, cx.const_bytes(section_contents));
            llvm::LLVMSetGlobalConstant(section_var, llvm::True);
            llvm::set_unnamed_address(section_var, llvm::UnnamedAddr::Global);
            llvm::set_linkage(section_var, llvm::Linkage::LinkOnceODRLinkage);
            // This should make sure that the whole section is not larger than
            // the string it contains. Otherwise we get a warning from GDB.
            llvm::LLVMSetAlignment(section_var, 1);
            // Make sure that the linker doesn't optimize the global away.
            cx.add_used_global(section_var);
            section_var
        }
    })
}

pub(crate) fn needs_gdb_debug_scripts_section(cx: &CodegenCx<'_, '_>) -> bool {
    let omit_gdb_pretty_printer_section =
        find_attr!(cx.tcx.hir_krate_attrs(), AttributeKind::OmitGdbPrettyPrinterSection);

    // We collect pretty printers transitively for all crates, so we make sure
    // that the section is only emitted for leaf crates.
    let embed_visualizers = cx.tcx.crate_types().iter().any(|&crate_type| match crate_type {
        CrateType::Executable | CrateType::Cdylib | CrateType::Staticlib | CrateType::Sdylib => {
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
        CrateType::Rlib | CrateType::Dylib => {
            // Don't embed pretty printers for these crate types; the compiler
            // can see the `#[debug_visualizer]` attributes when using the
            // library, and emitting `.debug_gdb_scripts` regardless would
            // break `#![omit_gdb_pretty_printer_section]`.
            false
        }
    });

    !omit_gdb_pretty_printer_section
        && cx.sess().opts.debuginfo != DebugInfo::None
        && cx.sess().target.emit_debug_gdb_scripts
        && embed_visualizers
}
