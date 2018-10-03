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
use rustc_codegen_ssa::debuginfo::FunctionDebugContext;

use llvm;
use llvm::debuginfo::{DIScope, DISubprogram};
use builder::Builder;
use rustc_codegen_ssa::interfaces::*;

use libc::c_uint;
use syntax_pos::{Span, Pos};
use value::Value;

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...).
pub fn set_source_location(
    debug_context: &FunctionDebugContext<&'ll DISubprogram>,
    bx: &Builder<'_, 'll, '_, &'ll Value>,
    scope: Option<&'ll DIScope>,
    span: Span,
) {
    let function_debug_context = match *debug_context {
        FunctionDebugContext::DebugInfoDisabled => return,
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            set_debug_location(bx, UnknownLocation);
            return;
        }
        FunctionDebugContext::RegularContext(ref data) => data
    };

    let dbg_loc = if function_debug_context.source_locations_enabled.get() {
        debug!("set_source_location: {}", bx.cx().sess().source_map().span_to_string(span));
        let loc = span_start(bx.cx(), span);
        InternalDebugLocation::new(scope.unwrap(), loc.line, loc.col.to_usize())
    } else {
        UnknownLocation
    };
    set_debug_location(bx, dbg_loc);
}


#[derive(Copy, Clone, PartialEq)]
pub enum InternalDebugLocation<'ll> {
    KnownLocation { scope: &'ll DIScope, line: usize, col: usize },
    UnknownLocation
}

impl InternalDebugLocation<'ll> {
    pub fn new(scope: &'ll DIScope, line: usize, col: usize) -> Self {
        KnownLocation {
            scope,
            line,
            col,
        }
    }
}

pub fn set_debug_location(
    bx: &Builder<'_, 'll, '_, &'ll Value>,
    debug_location: InternalDebugLocation<'ll>
) {
    let metadata_node = match debug_location {
        KnownLocation { scope, line, col } => {
            // For MSVC, set the column number to zero.
            // Otherwise, emit it. This mimics clang behaviour.
            // See discussion in https://github.com/rust-lang/rust/issues/42921
            let col_used =  if bx.cx().sess().target.target.options.is_like_msvc {
                UNKNOWN_COLUMN_NUMBER
            } else {
                col as c_uint
            };
            debug!("setting debug location to {} {}", line, col);

            unsafe {
                Some(llvm::LLVMRustDIBuilderCreateDebugLocation(
                    debug_context(bx.cx()).llcontext,
                    line as c_uint,
                    col_used,
                    scope,
                    None))
            }
        }
        UnknownLocation => {
            debug!("clearing debug location ");
            None
        }
    };

    unsafe {
        llvm::LLVMSetCurrentDebugLocation(bx.llbuilder, metadata_node);
    }
}
