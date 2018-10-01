// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Backend;
use super::HasCodegen;
use debuginfo::{FunctionDebugContext, MirDebugScope, VariableAccess, VariableKind};
use monomorphize::Instance;
use rustc::hir::def_id::CrateNum;
use rustc::mir;
use rustc::ty::{self, Ty};
use rustc_data_structures::indexed_vec::IndexVec;
use syntax::ast::Name;
use syntax_pos::{SourceFile, Span};

pub trait DebugInfoMethods<'tcx>: Backend<'tcx> {
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
        mir: &mir::Mir,
    ) -> FunctionDebugContext<Self::DIScope>;

    fn create_mir_scopes(
        &self,
        mir: &mir::Mir,
        debug_context: &FunctionDebugContext<Self::DIScope>,
    ) -> IndexVec<mir::SourceScope, MirDebugScope<Self::DIScope>>;
    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &SourceFile,
        defining_crate: CrateNum,
    ) -> Self::DIScope;
    fn debuginfo_finalize(&self);
}

pub trait DebugInfoBuilderMethods<'tcx>: HasCodegen<'tcx> {
    fn declare_local(
        &self,
        dbg_context: &FunctionDebugContext<Self::DIScope>,
        variable_name: Name,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_access: VariableAccess<'_, Self::Value>,
        variable_kind: VariableKind,
        span: Span,
    );
    fn set_source_location(
        &self,
        debug_context: &FunctionDebugContext<Self::DIScope>,
        scope: Option<Self::DIScope>,
        span: Span,
    );
    fn insert_reference_to_gdb_debug_scripts_section_global(&self);
}
