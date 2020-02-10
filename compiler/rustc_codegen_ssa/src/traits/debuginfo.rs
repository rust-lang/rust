use super::BackendTypes;
use crate::mir::debuginfo::{FunctionDebugContext, VariableKind};
use rustc_middle::mir;
use rustc_middle::ty::{Instance, Ty};
use rustc_span::{SourceFile, Span, Symbol};
use rustc_target::abi::call::FnAbi;
use rustc_target::abi::Size;

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
        mir: &mir::Body<'tcx>,
    ) -> Option<FunctionDebugContext<Self::DIScope, Self::DILocation>>;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<Self::Function>,
    ) -> Self::DIScope;

    fn dbg_loc(
        &self,
        scope: Self::DIScope,
        inlined_at: Option<Self::DILocation>,
        span: Span,
    ) -> Self::DILocation;

    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
    ) -> Self::DIScope;
    fn debuginfo_finalize(&self);

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn create_dbg_var(
        &self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_kind: VariableKind,
        span: Span,
    ) -> Self::DIVariable;
}

pub trait DebugInfoBuilderMethods: BackendTypes {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_var_addr(
        &mut self,
        dbg_var: Self::DIVariable,
        dbg_loc: Self::DILocation,
        variable_alloca: Self::Value,
        direct_offset: Size,
        // NB: each offset implies a deref (i.e. they're steps in a pointer chain).
        indirect_offsets: &[Size],
    );
    fn set_dbg_loc(&mut self, dbg_loc: Self::DILocation);
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self);
    fn set_var_name(&mut self, value: Self::Value, name: &str);
}
