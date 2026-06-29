use std::ops::Range;
use std::sync::Arc;

use gccjit::{Function, Location, RValue};
use rustc_abi::Size;
use rustc_codegen_ssa::mir::debuginfo::VariableKind;
use rustc_codegen_ssa::traits::{DebugInfoBuilderMethods, DebugInfoCodegenMethods};
use rustc_middle::ty::{ExistentialTraitRef, Instance, Ty};
use rustc_span::{BytePos, Pos, SourceFile, SourceFileAndLine, Span, Symbol};
use rustc_target::callconv::FnAbi;

use crate::builder::Builder;
use crate::context::CodegenCx;

pub(super) const UNKNOWN_LINE_NUMBER: u32 = 0;
pub(super) const UNKNOWN_COLUMN_NUMBER: u32 = 0;

impl<'a, 'gcc, 'tcx> DebugInfoBuilderMethods<'tcx> for Builder<'a, 'gcc, 'tcx> {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_var_addr(
        &mut self,
        _dbg_var: Self::DIVariable,
        _dbg_loc: Self::DILocation,
        _variable_alloca: Self::Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<Range<Size>>,
    ) {
        // FIXME(tempdragon): Not sure if this is correct, probably wrong but still keep it here.
        #[cfg(feature = "master")]
        _variable_alloca.set_location(_dbg_loc);
    }

    fn dbg_var_value(
        &mut self,
        _dbg_var: Self::DIVariable,
        _dbg_loc: Self::DILocation,
        _value: Self::Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<Range<Size>>,
    ) {
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        // FIXME(antoyo): insert reference to gdb debug scripts section global.
    }

    /// FIXME(tempdragon): Currently, this function is not yet implemented. It seems that the
    /// debug name and the mangled name should both be included in the LValues.
    /// Besides, a function to get the rvalue type(m_is_lvalue) should also be included.
    fn set_var_name(&mut self, _value: RValue<'gcc>, _name: &str) {}

    fn set_dbg_loc(&mut self, dbg_loc: Self::DILocation) {
        self.location = Some(dbg_loc);
    }

    fn clear_dbg_loc(&mut self) {
        self.location = None;
    }
}

/// A source code location used to generate debug information.
// FIXME(eddyb) rename this to better indicate it's a duplicate of
// `rustc_span::Loc` rather than `DILocation`, perhaps by making
// `lookup_char_pos` return the right information instead.
pub struct DebugLoc {
    /// Information about the original source file.
    pub file: Arc<SourceFile>,
    /// The (1-based) line number.
    pub line: u32,
    /// The (1-based) column number.
    pub col: u32,
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    /// Looks up debug source information about a `BytePos`.
    // FIXME(eddyb) rename this to better indicate it's a duplicate of
    // `lookup_char_pos` rather than `dbg_loc`, perhaps by making
    // `lookup_char_pos` return the right information instead.
    // Source of Origin: cg_llvm
    pub fn lookup_debug_loc(&self, pos: BytePos) -> DebugLoc {
        let (file, line, col) = match self.sess().source_map().lookup_line(pos) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.lines()[line];

                // Use 1-based indexing.
                let line = (line + 1) as u32;
                let col = (file.relative_position(pos) - line_pos).to_u32() + 1;

                (file, line, col)
            }
            Err(file) => (file, UNKNOWN_LINE_NUMBER, UNKNOWN_COLUMN_NUMBER),
        };

        // For MSVC, omit the column number.
        // Otherwise, emit it. This mimics clang behaviour.
        // See discussion in https://github.com/rust-lang/rust/issues/42921
        if self.sess().target.is_like_msvc {
            DebugLoc { file, line, col: UNKNOWN_COLUMN_NUMBER }
        } else {
            DebugLoc { file, line, col }
        }
    }
}

impl<'gcc, 'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn create_vtable_debuginfo(
        &self,
        _ty: Ty<'tcx>,
        _trait_ref: Option<ExistentialTraitRef<'tcx>>,
        _vtable: Self::Value,
    ) {
        // FIXME(antoyo)
    }

    fn dbg_create_lexical_block(
        &self,
        _pos: BytePos,
        _parent_scope: Self::DIScope,
    ) -> Self::DIScope {
    }

    fn dbg_location_clone_with_discriminator(
        &self,
        loc: Self::DILocation,
        _discriminator: u32,
    ) -> Option<Self::DILocation> {
        Some(loc)
    }

    fn extend_scope_to_file(
        &self,
        _scope_metadata: Self::DIScope,
        _file: &SourceFile,
    ) -> Self::DIScope {
        // FIXME(antoyo): implement.
    }

    fn debuginfo_finalize(&self) {
        self.context.set_debug_info(true)
    }

    fn create_dbg_var(
        &self,
        _variable_name: Symbol,
        _variable_type: Ty<'tcx>,
        _scope_metadata: Self::DIScope,
        _variable_kind: VariableKind,
        _span: Span,
    ) -> Self::DIVariable {
    }

    fn dbg_scope_fn(
        &self,
        _instance: Instance<'tcx>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _maybe_definition_llfn: Option<Function<'gcc>>,
    ) -> Self::DIScope {
        // FIXME(antoyo): implement.
    }

    fn dbg_loc(
        &self,
        _scope: Self::DIScope,
        _inlined_at: Option<Self::DILocation>,
        span: Span,
    ) -> Self::DILocation {
        let pos = span.lo();
        let DebugLoc { file, line, col } = self.lookup_debug_loc(pos);
        match file.name {
            rustc_span::FileName::Real(ref name) => self.context.new_location(
                name.path(rustc_span::RemapPathScopeComponents::DEBUGINFO).to_string_lossy(),
                line as i32,
                col as i32,
            ),
            _ => Location::null(),
        }
    }
}
