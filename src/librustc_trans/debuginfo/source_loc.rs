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
use super::metadata::{UNKNOWN_COLUMN_NUMBER};
use super::{FunctionDebugContext, DebugLoc};

use llvm;
use llvm::debuginfo::DIScope;
use builder::Builder;
use common::{CrateContext, FunctionContext};

use libc::c_uint;
use std::ptr;
use syntax_pos::Pos;

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...).
pub fn set_source_location(fcx: &FunctionContext,
                           builder: Option<&Builder>,
                           debug_loc: DebugLoc) {
    let builder = builder.map(|b| b.llbuilder);
    let function_debug_context = match fcx.debug_context {
        FunctionDebugContext::DebugInfoDisabled => return,
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            set_debug_location(fcx.ccx, builder, UnknownLocation);
            return;
        }
        FunctionDebugContext::RegularContext(box ref data) => data
    };

    if function_debug_context.source_location_override.get() {
        // Just ignore any attempts to set a new debug location while
        // the override is active.
        return;
    }

    let dbg_loc = if function_debug_context.source_locations_enabled.get() {
        let (scope, span) = match debug_loc {
            DebugLoc::ScopeAt(scope, span) => (scope, span),
            DebugLoc::None => {
                set_debug_location(fcx.ccx, builder, UnknownLocation);
                return;
            }
        };

        let cm = fcx.ccx.sess().codemap();
        if cm.is_valid_span(span) {
            debug!("set_source_location: {}",
                   fcx.ccx.sess().codemap().span_to_string(span));
            let loc = span_start(fcx.ccx, span);
            InternalDebugLocation::new(scope, loc.line, loc.col.to_usize())
        } else {
            // Span isn't invalid, just ignore the attempt to set a new debug
            // location
            return;
        }
    } else {
        UnknownLocation
    };
    set_debug_location(fcx.ccx, builder, dbg_loc);
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to translate a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is translated.
pub fn start_emitting_source_locations(fcx: &FunctionContext) {
    match fcx.debug_context {
        FunctionDebugContext::RegularContext(box ref data) => {
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

pub fn set_debug_location(cx: &CrateContext,
                          builder: Option<llvm::BuilderRef>,
                          debug_location: InternalDebugLocation) {
    if builder.is_none() {
        if debug_location == debug_context(cx).current_debug_location.get() {
            return;
        }
    }

    let metadata_node = match debug_location {
        KnownLocation { scope, line, .. } => {
            // Always set the column to zero like Clang and GCC
            let col = UNKNOWN_COLUMN_NUMBER;
            debug!("setting debug location to {} {}", line, col);

            unsafe {
                llvm::LLVMRustDIBuilderCreateDebugLocation(
                    debug_context(cx).llcontext,
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

    if builder.is_none() {
        debug_context(cx).current_debug_location.set(debug_location);
    }

    let builder = builder.unwrap_or_else(|| cx.raw_builder());
    unsafe {
        llvm::LLVMSetCurrentDebugLocation(builder, metadata_node);
    }
}
