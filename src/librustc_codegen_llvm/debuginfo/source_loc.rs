use super::metadata::UNKNOWN_COLUMN_NUMBER;
use super::utils::{debug_context, span_start};

use crate::common::CodegenCx;
use crate::llvm::debuginfo::DIScope;
use crate::llvm::{self, Value};
use rustc_codegen_ssa::traits::*;

use libc::c_uint;
use rustc_span::{Pos, Span};

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
