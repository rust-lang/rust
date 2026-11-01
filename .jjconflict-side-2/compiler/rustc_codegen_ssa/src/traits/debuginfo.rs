use std::ops::Range;

use rustc_abi::Size;
use rustc_middle::ty::{ExistentialTraitRef, Instance, Ty};
use rustc_span::{BytePos, SourceFile, Span, Symbol};
use rustc_target::callconv::FnAbi;

use super::BackendTypes;
use crate::mir::debuginfo::VariableKind;

pub trait DebugInfoCodegenMethods<'tcx>: BackendTypes {
    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<ExistentialTraitRef<'tcx>>,
        vtable: Self::Value,
    );
}

pub trait DebugInfoBuilderMethods<'tcx>: BackendTypes {
    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn dbg_scope_fn(
        &mut self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<Self::Function>,
    ) -> Self::DIScope;

    fn dbg_create_lexical_block(
        &mut self,
        pos: BytePos,
        parent_scope: Self::DIScope,
    ) -> Self::DIScope;

    fn dbg_location_clone_with_discriminator(
        &mut self,
        loc: Self::DILocation,
        discriminator: u32,
    ) -> Option<Self::DILocation>;

    fn dbg_loc(
        &mut self,
        scope: Self::DIScope,
        inlined_at: Option<Self::DILocation>,
        span: Span,
    ) -> Self::DILocation;

    fn extend_scope_to_file(
        &mut self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
    ) -> Self::DIScope;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `dbg`, `debug`, `debuginfo`, `debug_info` etc.).
    fn create_dbg_var(
        &mut self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_kind: VariableKind,
        span: Span,
    ) -> Self::DIVariable;

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
        // Byte range in the `dbg_var` covered by this fragment,
        // if this is a fragment of a composite `DIVariable`.
        fragment: &Option<Range<Size>>,
    );
    fn dbg_var_value(
        &mut self,
        dbg_var: Self::DIVariable,
        dbg_loc: Self::DILocation,
        value: Self::Value,
        direct_offset: Size,
        // NB: each offset implies a deref (i.e. they're steps in a pointer chain).
        indirect_offsets: &[Size],
        // Byte range in the `dbg_var` covered by this fragment,
        // if this is a fragment of a composite `DIVariable`.
        fragment: &Option<Range<Size>>,
    );
    fn set_dbg_loc(&mut self, dbg_loc: Self::DILocation);
    fn clear_dbg_loc(&mut self);
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self);
    fn set_var_name(&mut self, value: Self::Value, name: &str);

    /// Hook to allow move/copy operations to be annotated for profiling.
    ///
    /// The `instance` parameter should be the monomorphized instance of the
    /// `compiler_move` or `compiler_copy` function with the actual type and size.
    ///
    /// Default implementation does no annotation (just executes the closure).
    fn with_move_annotation<R>(
        &mut self,
        _instance: Instance<'tcx>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        f(self)
    }
}
