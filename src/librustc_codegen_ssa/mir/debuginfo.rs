use rustc::hir::def_id::CrateNum;
use rustc::mir;
use crate::traits::*;

use syntax_pos::{DUMMY_SP, BytePos, Span};

use super::FunctionCx;

pub enum FunctionDebugContext<D> {
    RegularContext(FunctionDebugContextData<D>),
    DebugInfoDisabled,
    FunctionWithoutDebugInfo,
}

impl<D> FunctionDebugContext<D> {
    pub fn get_ref(&self, span: Span) -> &FunctionDebugContextData<D> {
        match *self {
            FunctionDebugContext::RegularContext(ref data) => data,
            FunctionDebugContext::DebugInfoDisabled => {
                span_bug!(
                    span,
                    "debuginfo: Error trying to access FunctionDebugContext \
                     although debug info is disabled!",
                );
            }
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                span_bug!(
                    span,
                    "debuginfo: Error trying to access FunctionDebugContext \
                     for function that should be ignored by debug info!",
                );
            }
        }
    }
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to codegen a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is codegened.
pub fn start_emitting_source_locations<D>(dbg_context: &mut FunctionDebugContext<D>) {
    match *dbg_context {
        FunctionDebugContext::RegularContext(ref mut data) => {
            data.source_locations_enabled = true;
        },
        _ => { /* safe to ignore */ }
    }
}

pub struct FunctionDebugContextData<D> {
    pub fn_metadata: D,
    pub source_locations_enabled: bool,
    pub defining_crate: CrateNum,
}

pub enum VariableAccess<'a, V> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: V },
    // The llptr given is an alloca containing the start of some pointer chain
    // leading to the variable's content.
    IndirectVariable { alloca: V, address_operations: &'a [i64] }
}

pub enum VariableKind {
    ArgumentVariable(usize /*index*/),
    LocalVariable,
}


#[derive(Clone, Copy, Debug)]
pub struct DebugScope<D> {
    pub scope_metadata: Option<D>,
    // Start and end offsets of the file to which this DIScope belongs.
    // These are used to quickly determine whether some span refers to the same file.
    pub file_start_pos: BytePos,
    pub file_end_pos: BytePos,
}

impl<D> DebugScope<D> {
    pub fn is_valid(&self) -> bool {
        !self.scope_metadata.is_none()
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn set_debug_loc(
        &mut self,
        bx: &mut Bx,
        source_info: mir::SourceInfo
    ) {
        let (scope, span) = self.debug_loc(source_info);
        bx.set_source_location(&mut self.debug_context, scope, span);
    }

    pub fn debug_loc(&self, source_info: mir::SourceInfo) -> (Option<Bx::DIScope>, Span) {
        // Bail out if debug info emission is not enabled.
        match self.debug_context {
            FunctionDebugContext::DebugInfoDisabled |
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                return (self.scopes[source_info.scope].scope_metadata, source_info.span);
            }
            FunctionDebugContext::RegularContext(_) =>{}
        }

        // In order to have a good line stepping behavior in debugger, we overwrite debug
        // locations of macro expansions with that of the outermost expansion site
        // (unless the crate is being compiled with `-Z debug-macros`).
        if !source_info.span.from_expansion() ||
           self.cx.sess().opts.debugging_opts.debug_macros {
            let scope = self.scope_metadata_for_loc(source_info.scope, source_info.span.lo());
            (scope, source_info.span)
        } else {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            let span = syntax_pos::hygiene::walk_chain(source_info.span, self.mir.span.ctxt());
            let scope = self.scope_metadata_for_loc(source_info.scope, span.lo());
            // Use span of the outermost expansion site, while keeping the original lexical scope.
            (scope, span)
        }
    }

    // DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    // it may so happen that the current span belongs to a different file than the DIScope
    // corresponding to span's containing source scope.  If so, we need to create a DIScope
    // "extension" into that file.
    fn scope_metadata_for_loc(&self, scope_id: mir::SourceScope, pos: BytePos)
                              -> Option<Bx::DIScope> {
        let scope_metadata = self.scopes[scope_id].scope_metadata;
        if pos < self.scopes[scope_id].file_start_pos ||
           pos >= self.scopes[scope_id].file_end_pos {
            let sm = self.cx.sess().source_map();
            let defining_crate = self.debug_context.get_ref(DUMMY_SP).defining_crate;
            Some(self.cx.extend_scope_to_file(
                scope_metadata.unwrap(),
                &sm.lookup_char_pos(pos).file,
                defining_crate
            ))
        } else {
            scope_metadata
        }
    }
}
