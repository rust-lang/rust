use syntax_pos::{BytePos, Span};
use rustc::hir::def_id::CrateNum;

pub mod type_names;

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
pub struct MirDebugScope<D> {
    pub scope_metadata: Option<D>,
    // Start and end offsets of the file to which this DIScope belongs.
    // These are used to quickly determine whether some span refers to the same file.
    pub file_start_pos: BytePos,
    pub file_end_pos: BytePos,
}

impl<D> MirDebugScope<D> {
    pub fn is_valid(&self) -> bool {
        !self.scope_metadata.is_none()
    }
}
