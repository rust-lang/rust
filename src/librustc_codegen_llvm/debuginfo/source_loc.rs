use self::InternalDebugLocation::*;

use super::utils::{debug_context, span_start};
use super::metadata::UNKNOWN_COLUMN_NUMBER;
use rustc_codegen_ssa::debuginfo::FunctionDebugContext;

use crate::llvm;
use crate::llvm::debuginfo::DIScope;
use crate::builder::Builder;
use rustc_codegen_ssa::traits::*;

use libc::c_uint;
use syntax_pos::{Span, Pos};

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...).
pub fn set_source_location<D>(
    debug_context: &FunctionDebugContext<D>,
    bx: &Builder<'_, 'll, '_>,
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

    let dbg_loc = if function_debug_context.source_locations_enabled {
        debug!("set_source_location: {}", bx.sess().source_map().span_to_string(span));
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
    bx: &Builder<'_, 'll, '_>,
    debug_location: InternalDebugLocation<'ll>
) {
    let metadata_node = match debug_location {
        KnownLocation { scope, line, col } => {
            // For MSVC, set the column number to zero.
            // Otherwise, emit it. This mimics clang behaviour.
            // See discussion in https://github.com/rust-lang/rust/issues/42921
            let col_used =  if bx.sess().target.target.options.is_like_msvc {
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
