// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{Ty, FnSig};
use super::backend::Backend;
use super::builder::HasCodegen;
use rustc::mir;
use monomorphize::Instance;
use debuginfo::{FunctionDebugContext, MirDebugScope, VariableAccess, VariableKind};
use rustc_data_structures::indexed_vec::IndexVec;
use syntax_pos;
use rustc::hir::def_id::CrateNum;
use syntax::ast::Name;

pub trait DebugInfoMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    type DIScope : 'll + Copy;

    fn create_vtable_metadata(
        &self,
        ty: Ty<'tcx>,
        vtable: Self::Value,
    );

    /// Creates the function-specific debug context.
    ///
    /// Returns the FunctionDebugContext for the function which holds state needed
    /// for debug info creation. The function may also return another variant of the
    /// FunctionDebugContext enum which indicates why no debuginfo should be created
    /// for the function.
    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        sig: FnSig<'tcx>,
        llfn: Self::Value,
        mir: &mir::Mir,
    ) -> FunctionDebugContext<'ll>;
    
    fn create_mir_scopes(
        &self,
        mir: &mir::Mir,
        debug_context: &FunctionDebugContext<'ll>,
    ) -> IndexVec<mir::SourceScope, MirDebugScope<Self::DIScope>>;
    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &syntax_pos::SourceFile,
        defining_crate: CrateNum,
    ) -> Self::DIScope;
}

pub trait DebugInfoBuilderMethods<'a, 'll: 'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx> {
    fn declare_local(
        &self,
        dbg_context: &FunctionDebugContext<'ll>,
        variable_name: Name,
        variable_type: Ty<'tcx>,
        scope_metadata: <Self::CodegenCx as DebugInfoMethods<'ll, 'tcx>>::DIScope,
        variable_access: VariableAccess<'_, <Self::CodegenCx as Backend<'ll>>::Value>,
        variable_kind: VariableKind,
        span: syntax_pos::Span,
    );
    fn set_source_location(
        &self,
        debug_context: &FunctionDebugContext<'ll>,
        scope: Option<<Self::CodegenCx as DebugInfoMethods<'ll, 'tcx>>::DIScope>,
        span: syntax_pos::Span,
    );
    fn insert_reference_to_gdb_debug_scripts_section_global(&self);
}
