// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::InternalDebugLocation::*;

use super::utils::{debug_context, span_start};
use super::metadata::UNKNOWN_COLUMN_NUMBER;
use super::FunctionDebugContext;

use llvm;
use llvm::debuginfo::DIScope;
use builder::Builder;

use libc::c_uint;
use std::ptr;
use syntax_pos::{Span, Pos};

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...).
pub fn set_source_location(
    debug_context: &FunctionDebugContext, builder: &Builder, scope: DIScope, span: Span
) {
    let function_debug_context = match *debug_context {
        FunctionDebugContext::DebugInfoDisabled => return,
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            set_debug_location(builder, UnknownLocation);
            return;
        }
        FunctionDebugContext::RegularContext(ref data) => data
    };

    let dbg_loc = if function_debug_context.source_locations_enabled.get() {
        debug!("set_source_location: {}", builder.sess().codemap().span_to_string(span));
        let loc = span_start(builder.ccx, span);
        InternalDebugLocation::new(scope, loc.line, loc.col.to_usize())
    } else {
        UnknownLocation
    };
    set_debug_location(builder, dbg_loc);
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to translate a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is translated.
pub fn start_emitting_source_locations(dbg_context: &FunctionDebugContext) {
    match *dbg_context {
        FunctionDebugContext::RegularContext(ref data) => {
            data.source_locations_enabled.set(true)
        },
        _ => { /* safe to ignore */ }
    }
}


#[derive(Copy, Clone, PartialEq)]
pub enum InternalDebugLocation {
    KnownLocation { scope: DIScope, line: usize, col: usize },
    UnknownLocation
}

impl InternalDebugLocation {
    pub fn new(scope: DIScope, line: usize, col: usize) -> InternalDebugLocation {
        KnownLocation {
            scope: scope,
            line: line,
            col: col,
        }
    }
}

pub fn set_debug_location(builder: &Builder, debug_location: InternalDebugLocation) {
    let metadata_node = match debug_location {
        KnownLocation { scope, line, .. } => {
            // Always set the column to zero like Clang and GCC
            let col = UNKNOWN_COLUMN_NUMBER;
            debug!("setting debug location to {} {}", line, col);

            unsafe {
                llvm::LLVMRustDIBuilderCreateDebugLocation(
                    debug_context(builder.ccx).llcontext,
                    line as c_uint,
                    col as c_uint,
                    scope,
                    ptr::null_mut())
            }
        }
        UnknownLocation => {
            debug!("clearing debug location ");
            ptr::null_mut()
        }
    };

    unsafe {
        llvm::LLVMSetCurrentDebugLocation(builder.llbuilder, metadata_node);
    }
}
