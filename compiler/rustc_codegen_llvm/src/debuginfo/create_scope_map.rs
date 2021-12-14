use super::metadata::file_metadata;
use super::utils::DIB;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext};
use rustc_codegen_ssa::traits::*;

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DILocation, DIScope};
use rustc_middle::mir::{Body, SourceScope};
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::{self, Instance};
use rustc_session::config::DebugInfo;

use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;

/// Produces DIScope DIEs for each MIR Scope which has variables defined in it.
// FIXME(eddyb) almost all of this should be in `rustc_codegen_ssa::mir::debuginfo`.
pub fn compute_mir_scopes<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
    mir: &Body<'tcx>,
    fn_dbg_scope: &'ll DIScope,
    debug_context: &mut FunctionDebugContext<&'ll DIScope, &'ll DILocation>,
) {
    // Find all the scopes with variables defined in them.
    let mut has_variables = BitSet::new_empty(mir.source_scopes.len());

    // Only consider variables when they're going to be emitted.
    // FIXME(eddyb) don't even allocate `has_variables` otherwise.
    if cx.sess().opts.debuginfo == DebugInfo::Full {
        // FIXME(eddyb) take into account that arguments always have debuginfo,
        // irrespective of their name (assuming full debuginfo is enabled).
        // NOTE(eddyb) actually, on second thought, those are always in the
        // function scope, which always exists.
        for var_debug_info in &mir.var_debug_info {
            has_variables.insert(var_debug_info.source_info.scope);
        }
    }

    // Instantiate all scopes.
    for idx in 0..mir.source_scopes.len() {
        let scope = SourceScope::new(idx);
        make_mir_scope(cx, instance, mir, fn_dbg_scope, &has_variables, debug_context, scope);
    }
}

fn make_mir_scope<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
    mir: &Body<'tcx>,
    fn_dbg_scope: &'ll DIScope,
    has_variables: &BitSet<SourceScope>,
    debug_context: &mut FunctionDebugContext<&'ll DIScope, &'ll DILocation>,
    scope: SourceScope,
) {
    if debug_context.scopes[scope].dbg_scope.is_some() {
        return;
    }

    let scope_data = &mir.source_scopes[scope];
    let parent_scope = if let Some(parent) = scope_data.parent_scope {
        make_mir_scope(cx, instance, mir, fn_dbg_scope, has_variables, debug_context, parent);
        debug_context.scopes[parent]
    } else {
        // The root is the function itself.
        let loc = cx.lookup_debug_loc(mir.span.lo());
        debug_context.scopes[scope] = DebugScope {
            dbg_scope: Some(fn_dbg_scope),
            inlined_at: None,
            file_start_pos: loc.file.start_pos,
            file_end_pos: loc.file.end_pos,
        };
        return;
    };

    if !has_variables.contains(scope) && scope_data.inlined.is_none() {
        // Do not create a DIScope if there are no variables defined in this
        // MIR `SourceScope`, and it's not `inlined`, to avoid debuginfo bloat.
        debug_context.scopes[scope] = parent_scope;
        return;
    }

    let loc = cx.lookup_debug_loc(scope_data.span.lo());
    let file_metadata = file_metadata(cx, &loc.file);

    let dbg_scope = match scope_data.inlined {
        Some((callee, _)) => {
            // FIXME(eddyb) this would be `self.monomorphize(&callee)`
            // if this is moved to `rustc_codegen_ssa::mir::debuginfo`.
            let callee = cx.tcx.subst_and_normalize_erasing_regions(
                instance.substs,
                ty::ParamEnv::reveal_all(),
                callee,
            );
            let callee_fn_abi = cx.fn_abi_of_instance(callee, ty::List::empty());
            cx.dbg_scope_fn(callee, callee_fn_abi, None)
        }
        None => unsafe {
            llvm::LLVMRustDIBuilderCreateLexicalBlock(
                DIB(cx),
                parent_scope.dbg_scope.unwrap(),
                file_metadata,
                loc.line,
                loc.col,
            )
        },
    };

    let inlined_at = scope_data.inlined.map(|(_, callsite_span)| {
        // FIXME(eddyb) this doesn't account for the macro-related
        // `Span` fixups that `rustc_codegen_ssa::mir::debuginfo` does.
        let callsite_scope = parent_scope.adjust_dbg_scope_for_span(cx, callsite_span);
        cx.dbg_loc(callsite_scope, parent_scope.inlined_at, callsite_span)
    });

    debug_context.scopes[scope] = DebugScope {
        dbg_scope: Some(dbg_scope),
        inlined_at: inlined_at.or(parent_scope.inlined_at),
        file_start_pos: loc.file.start_pos,
        file_end_pos: loc.file.end_pos,
    };
}
