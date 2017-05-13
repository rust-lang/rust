// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// .debug_gdb_scripts binary section.

use llvm;

use common::{C_bytes, CrateContext, C_i32};
use builder::Builder;
use declare;
use type_::Type;
use session::config::NoDebugInfo;

use std::ptr;
use syntax::attr;


/// Inserts a side-effect free instruction sequence that makes sure that the
/// .debug_gdb_scripts global is referenced, so it isn't removed by the linker.
pub fn insert_reference_to_gdb_debug_scripts_section_global(ccx: &CrateContext, builder: &Builder) {
    if needs_gdb_debug_scripts_section(ccx) {
        let gdb_debug_scripts_section_global = get_or_insert_gdb_debug_scripts_section_global(ccx);
        // Load just the first byte as that's all that's necessary to force
        // LLVM to keep around the reference to the global.
        let indices = [C_i32(ccx, 0), C_i32(ccx, 0)];
        let element = builder.inbounds_gep(gdb_debug_scripts_section_global, &indices);
        let volative_load_instruction = builder.volatile_load(element);
        unsafe {
            llvm::LLVMSetAlignment(volative_load_instruction, 1);
        }
    }
}

/// Allocates the global variable responsible for the .debug_gdb_scripts binary
/// section.
pub fn get_or_insert_gdb_debug_scripts_section_global(ccx: &CrateContext)
                                                  -> llvm::ValueRef {
    let c_section_var_name = "__rustc_debug_gdb_scripts_section__\0";
    let section_var_name = &c_section_var_name[..c_section_var_name.len()-1];

    let section_var = unsafe {
        llvm::LLVMGetNamedGlobal(ccx.llmod(),
                                 c_section_var_name.as_ptr() as *const _)
    };

    if section_var == ptr::null_mut() {
        let section_name = b".debug_gdb_scripts\0";
        let section_contents = b"\x01gdb_load_rust_pretty_printers.py\0";

        unsafe {
            let llvm_type = Type::array(&Type::i8(ccx),
                                        section_contents.len() as u64);

            let section_var = declare::define_global(ccx, section_var_name,
                                                     llvm_type).unwrap_or_else(||{
                bug!("symbol `{}` is already defined", section_var_name)
            });
            llvm::LLVMSetSection(section_var, section_name.as_ptr() as *const _);
            llvm::LLVMSetInitializer(section_var, C_bytes(ccx, section_contents));
            llvm::LLVMSetGlobalConstant(section_var, llvm::True);
            llvm::LLVMSetUnnamedAddr(section_var, llvm::True);
            llvm::LLVMRustSetLinkage(section_var, llvm::Linkage::LinkOnceODRLinkage);
            // This should make sure that the whole section is not larger than
            // the string it contains. Otherwise we get a warning from GDB.
            llvm::LLVMSetAlignment(section_var, 1);
            section_var
        }
    } else {
        section_var
    }
}

pub fn needs_gdb_debug_scripts_section(ccx: &CrateContext) -> bool {
    let omit_gdb_pretty_printer_section =
        attr::contains_name(&ccx.tcx().map.krate_attrs(),
                            "omit_gdb_pretty_printer_section");

    !omit_gdb_pretty_printer_section &&
    !ccx.sess().target.target.options.is_like_osx &&
    !ccx.sess().target.target.options.is_like_windows &&
    ccx.sess().opts.debuginfo != NoDebugInfo
}
