// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Module-Internal debug info creation functions.

use super::utils::{span_start, DIB};
use super::metadata::{type_metadata, file_metadata};

use super::{set_debug_location, DW_TAG_auto_variable, DW_TAG_arg_variable};
use super::VariableKind::{self, ArgumentVariable, CapturedVariable, LocalVariable};
use super::VariableAccess::{self, DirectVariable, IndirectVariable};
use super::InternalDebugLocation::{self, UnknownLocation};

use llvm;
use llvm::debuginfo::{DIScope, DIBuilderRef, DIDescriptor, DIArray};

use trans;
use trans::common::{CrateContext, Block};
use middle::ty::Ty;
use session::config;

use libc::c_uint;
use std::ffi::CString;
use syntax::codemap::{Span, Pos};
use syntax::ast;
use syntax::parse::token;

pub fn is_node_local_to_unit(cx: &CrateContext, node_id: ast::NodeId) -> bool
{
    // The is_local_to_unit flag indicates whether a function is local to the
    // current compilation unit (i.e. if it is *static* in the C-sense). The
    // *reachable* set should provide a good approximation of this, as it
    // contains everything that might leak out of the current crate (by being
    // externally visible or by being inlined into something externally
    // visible). It might better to use the `exported_items` set from
    // `driver::CrateAnalysis` in the future, but (atm) this set is not
    // available in the translation pass.
    !cx.reachable().contains(&node_id)
}

#[allow(non_snake_case)]
pub fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len() as u32)
    };
}

pub fn declare_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 variable_name: ast::Name,
                                 variable_type: Ty<'tcx>,
                                 scope_metadata: DIScope,
                                 variable_access: VariableAccess,
                                 variable_kind: VariableKind,
                                 span: Span) {
    let cx: &CrateContext = bcx.ccx();

    let filename = span_start(cx, span).file.name.clone();
    let file_metadata = file_metadata(cx, &filename[..]);

    let name = token::get_name(variable_name);
    let loc = span_start(cx, span);
    let type_metadata = type_metadata(cx, variable_type, span);

    let (argument_index, dwarf_tag) = match variable_kind {
        ArgumentVariable(index) => (index as c_uint, DW_TAG_arg_variable),
        LocalVariable    |
        CapturedVariable => (0, DW_TAG_auto_variable)
    };

    let name = CString::new(name.as_bytes()).unwrap();
    match (variable_access, &[][..]) {
        (DirectVariable { alloca }, address_operations) |
        (IndirectVariable {alloca, address_operations}, _) => {
            let metadata = unsafe {
                llvm::LLVMDIBuilderCreateVariable(
                    DIB(cx),
                    dwarf_tag,
                    scope_metadata,
                    name.as_ptr(),
                    file_metadata,
                    loc.line as c_uint,
                    type_metadata,
                    cx.sess().opts.optimize != config::No,
                    0,
                    address_operations.as_ptr(),
                    address_operations.len() as c_uint,
                    argument_index)
            };
            set_debug_location(cx, InternalDebugLocation::new(scope_metadata,
                                                      loc.line,
                                                      loc.col.to_usize()));
            unsafe {
                let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(
                    DIB(cx),
                    alloca,
                    metadata,
                    address_operations.as_ptr(),
                    address_operations.len() as c_uint,
                    bcx.llbb);

                llvm::LLVMSetInstDebugLocation(trans::build::B(bcx).llbuilder, instr);
            }
        }
    }

    match variable_kind {
        ArgumentVariable(_) | CapturedVariable => {
            assert!(!bcx.fcx
                        .debug_context
                        .get_ref(cx, span)
                        .source_locations_enabled
                        .get());
            set_debug_location(cx, UnknownLocation);
        }
        _ => { /* nothing to do */ }
    }
}

