// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::FunctionDebugContext;
use super::metadata::file_metadata;
use super::utils::{DIB, span_start};

use llvm;
use llvm::debuginfo::{DIScope, DISubprogram};
use common::{CrateContext, FunctionContext};
use rustc::mir::repr::{Mir, VisibilityScope};

use libc::c_uint;
use std::ptr;

use syntax_pos::Pos;

use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

/// Produce DIScope DIEs for each MIR Scope which has variables defined in it.
/// If debuginfo is disabled, the returned vector is empty.
pub fn create_mir_scopes(fcx: &FunctionContext) -> IndexVec<VisibilityScope, DIScope> {
    let mir = fcx.mir.clone().expect("create_mir_scopes: missing MIR for fn");
    let mut scopes = IndexVec::from_elem(ptr::null_mut(), &mir.visibility_scopes);

    let fn_metadata = match fcx.debug_context {
        FunctionDebugContext::RegularContext(box ref data) => data.fn_metadata,
        FunctionDebugContext::DebugInfoDisabled |
        FunctionDebugContext::FunctionWithoutDebugInfo => {
            return scopes;
        }
    };

    // Find all the scopes with variables defined in them.
    let mut has_variables = BitVector::new(mir.visibility_scopes.len());
    for var in &mir.var_decls {
        has_variables.insert(var.source_info.scope.index());
    }

    // Instantiate all scopes.
    for idx in 0..mir.visibility_scopes.len() {
        let scope = VisibilityScope::new(idx);
        make_mir_scope(fcx.ccx, &mir, &has_variables, fn_metadata, scope, &mut scopes);
    }

    scopes
}

fn make_mir_scope(ccx: &CrateContext,
                  mir: &Mir,
                  has_variables: &BitVector,
                  fn_metadata: DISubprogram,
                  scope: VisibilityScope,
                  scopes: &mut IndexVec<VisibilityScope, DIScope>) {
    if !scopes[scope].is_null() {
        return;
    }

    let scope_data = &mir.visibility_scopes[scope];
    let parent_scope = if let Some(parent) = scope_data.parent_scope {
        make_mir_scope(ccx, mir, has_variables, fn_metadata, parent, scopes);
        scopes[parent]
    } else {
        // The root is the function itself.
        scopes[scope] = fn_metadata;
        return;
    };

    if !has_variables.contains(scope.index()) {
        // Do not create a DIScope if there are no variables
        // defined in this MIR Scope, to avoid debuginfo bloat.

        // However, we don't skip creating a nested scope if
        // our parent is the root, because we might want to
        // put arguments in the root and not have shadowing.
        if parent_scope != fn_metadata {
            scopes[scope] = parent_scope;
            return;
        }
    }

    let loc = span_start(ccx, scope_data.span);
    scopes[scope] = unsafe {
    let file_metadata = file_metadata(ccx, &loc.file.name, &loc.file.abs_path);
        llvm::LLVMRustDIBuilderCreateLexicalBlock(
            DIB(ccx),
            parent_scope,
            file_metadata,
            loc.line as c_uint,
            loc.col.to_usize() as c_uint)
    };
}
