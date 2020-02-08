use super::metadata::UNKNOWN_COLUMN_NUMBER;
use super::utils::{debug_context, span_start};
use rustc_codegen_ssa::mir::debuginfo::FunctionDebugContext;

use crate::builder::Builder;
use crate::common::CodegenCx;
use crate::llvm::debuginfo::DIScope;
use crate::llvm::{self, Value};
use log::debug;
use rustc_codegen_ssa::traits::*;

use libc::c_uint;
use rustc_span::{Pos, Span};

/// Sets the current debug location at the beginning of the span.
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...).
pub fn set_source_location<D>(
    debug_context: &FunctionDebugContext<D>,
    bx: &Builder<'_, 'll, '_>,
    scope: &'ll DIScope,
    span: Span,
) {
    let dbg_loc = if debug_context.source_locations_enabled {
        debug!("set_source_location: {}", bx.sess().source_map().span_to_string(span));
        Some(bx.cx().create_debug_loc(scope, span))
    } else {
        None
    };

    unsafe {
        llvm::LLVMSetCurrentDebugLocation(bx.llbuilder, dbg_loc);
    }
}

impl CodegenCx<'ll, '_> {
    pub fn create_debug_loc(&self, scope: &'ll DIScope, span: Span) -> &'ll Value {
        let loc = span_start(self, span);

        // For MSVC, set the column number to zero.
        // Otherwise, emit it. This mimics clang behaviour.
        // See discussion in https://github.com/rust-lang/rust/issues/42921
        let col_used = if self.sess().target.target.options.is_like_msvc {
            UNKNOWN_COLUMN_NUMBER
        } else {
            loc.col.to_usize() as c_uint
        };

        unsafe {
            llvm::LLVMRustDIBuilderCreateDebugLocation(
                debug_context(self).llcontext,
                loc.line as c_uint,
                col_used,
                scope,
                None,
            )
        }
    }
}
