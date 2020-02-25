use super::metadata::{UNKNOWN_COLUMN_NUMBER, UNKNOWN_LINE_NUMBER};
use super::utils::debug_context;

use crate::common::CodegenCx;
use crate::llvm::debuginfo::DIScope;
use crate::llvm::{self, Value};
use rustc_codegen_ssa::traits::*;

use rustc_data_structures::sync::Lrc;
use rustc_span::{BytePos, Pos, SourceFile, SourceFileAndLine, Span};

/// A source code location used to generate debug information.
pub struct DebugLoc {
    /// Information about the original source file.
    pub file: Lrc<SourceFile>,
    /// The (1-based) line number.
    pub line: Option<u32>,
    /// The (1-based) column number.
    pub col: Option<u32>,
}

impl CodegenCx<'ll, '_> {
    /// Looks up debug source information about a `BytePos`.
    pub fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.line_begin_pos(pos);

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (pos - line_pos).to_u32() + 1;

                (file, Some(line), Some(col))
            }
            Err(file) => (file, None, None),
        };

        // For MSVC, omit the column number.
        // Otherwise, emit it. This mimics clang behaviour.
        // See discussion in https://github.com/rust-lang/rust/issues/42921
        if self.sess().target.target.options.is_like_msvc {
            DebugLoc { file, line, col: None }
        } else {
            DebugLoc { file, line, col }
        }
    }

    pub fn create_debug_loc(&self, scope: &'ll DIScope, span: Span) -> &'ll Value {
        let DebugLoc { line, col, .. } = self.lookup_debug_loc(span.lo());

        unsafe {
            llvm::LLVMRustDIBuilderCreateDebugLocation(
                debug_context(self).llcontext,
                line.unwrap_or(UNKNOWN_LINE_NUMBER),
                col.unwrap_or(UNKNOWN_COLUMN_NUMBER),
                scope,
                None,
            )
        }
    }
}
