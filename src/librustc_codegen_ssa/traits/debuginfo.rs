use super::BackendTypes;
use crate::mir::debuginfo::{FunctionDebugContext, VariableKind};
use rustc::mir;
use rustc::ty::layout::Size;
use rustc::ty::{Instance, Ty};
use rustc_hir::def_id::CrateNum;
use rustc_span::{SourceFile, Span};
use rustc_target::abi::call::FnAbi;
use syntax::ast::Name;

pub trait DebugInfoMethods<'tcx>: BackendTypes {
    fn create_vtable_metadata(&self, ty: Ty<'tcx>, vtable: Self::Value);

    /// Creates the function-specific debug context.
    ///
    /// Returns the FunctionDebugContext for the function which holds state needed
    /// for debug info creation, if it is enabled.
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: Self::Function,
        mir: &mir::Body<'_>,
    ) -> Option<FunctionDebugContext<Self::DIScope>>;

    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
        defining_crate: CrateNum,
    ) -> Self::DIScope;
    fn debuginfo_finalize(&self);
}

pub trait DebugInfoBuilderMethods<'tcx>: BackendTypes {
    fn declare_local(
        &mut self,
        dbg_context: &FunctionDebugContext<Self::DIScope>,
        variable_name: Name,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_alloca: Self::Value,
        direct_offset: Size,
        // NB: each offset implies a deref (i.e. they're steps in a pointer chain).
        indirect_offsets: &[Size],
        variable_kind: VariableKind,
        span: Span,
    );
    fn set_source_location(
        &mut self,
        debug_context: &mut FunctionDebugContext<Self::DIScope>,
        scope: Self::DIScope,
        span: Span,
    );
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self);
    fn set_var_name(&mut self, value: Self::Value, name: &str);
}
