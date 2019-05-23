use super::BackendTypes;
use crate::debuginfo::{FunctionDebugContext, MirDebugScope, VariableAccess, VariableKind};
use rustc::hir::def_id::CrateNum;
use rustc::mir;
use rustc::ty::{self, Ty, Instance};
use rustc_data_structures::indexed_vec::IndexVec;
use syntax::ast::Name;
use syntax_pos::{SourceFile, Span};

pub trait DebugInfoMethods<'tcx>: BackendTypes {
    fn create_vtable_metadata(&self, ty: Ty<'tcx>, vtable: Self::Value);

    /// Creates the function-specific debug context.
    ///
    /// Returns the FunctionDebugContext for the function which holds state needed
    /// for debug info creation. The function may also return another variant of the
    /// FunctionDebugContext enum which indicates why no debuginfo should be created
    /// for the function.
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        sig: ty::FnSig<'tcx>,
        llfn: Self::Value,
        mir: &mir::Body<'_>,
    ) -> FunctionDebugContext<Self::DIScope>;

    fn create_mir_scopes(
        &self,
        mir: &mir::Body<'_>,
        debug_context: &mut FunctionDebugContext<Self::DIScope>,
    ) -> IndexVec<mir::SourceScope, MirDebugScope<Self::DIScope>>;
    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
        defining_crate: CrateNum,
    ) -> Self::DIScope;
    fn debuginfo_finalize(&self);
    fn debuginfo_upvar_ops_sequence(&self, byte_offset_of_var_in_env: u64) -> [i64; 4];
}

pub trait DebugInfoBuilderMethods<'tcx>: BackendTypes {
    fn declare_local(
        &mut self,
        dbg_context: &FunctionDebugContext<Self::DIScope>,
        variable_name: Name,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_access: VariableAccess<'_, Self::Value>,
        variable_kind: VariableKind,
        span: Span,
    );
    fn set_source_location(
        &mut self,
        debug_context: &mut FunctionDebugContext<Self::DIScope>,
        scope: Option<Self::DIScope>,
        span: Span,
    );
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self);
    fn set_value_name(&mut self, value: Self::Value, name: &str);
}
