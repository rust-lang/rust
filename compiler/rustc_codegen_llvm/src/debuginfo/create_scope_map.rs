use std::collections::hash_map::Entry;

use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{Body, SourceScope};
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv};
use rustc_middle::ty::{self, Instance};
use rustc_session::config::DebugInfo;
use rustc_span::{BytePos, DUMMY_SP, hygiene};

use super::metadata::file_metadata;
use super::utils::DIB;
use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DILocation, DIScope};

/// Produces DIScope DIEs for each MIR Scope which has variables defined in it.
// FIXME(eddyb) almost all of this should be in `rustc_codegen_ssa::mir::debuginfo`.
pub(crate) fn compute_mir_scopes<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
    mir: &Body<'tcx>,
    debug_context: &mut FunctionDebugContext<'tcx, &'ll DIScope, &'ll DILocation>,
) {
    // Find all scopes with variables defined in them.
    let variables = if cx.sess().opts.debuginfo == DebugInfo::Full {
        let mut vars = DenseBitSet::new_empty(mir.source_scopes.len());
        // FIXME(eddyb) take into account that arguments always have debuginfo,
        // irrespective of their name (assuming full debuginfo is enabled).
        // NOTE(eddyb) actually, on second thought, those are always in the
        // function scope, which always exists.
        for var_debug_info in &mir.var_debug_info {
            vars.insert(var_debug_info.source_info.scope);
        }
        Some(vars)
    } else {
        // Nothing to emit, of course.
        None
    };
    let mut instantiated = DenseBitSet::new_empty(mir.source_scopes.len());
    let mut discriminators = FxHashMap::default();
    // Instantiate all scopes.
    for scope in mir.source_scopes.indices() {
        make_mir_scope(
            cx,
            instance,
            mir,
            &variables,
            debug_context,
            &mut instantiated,
            &mut discriminators,
            scope,
        );
    }
    assert!(instantiated.count() == mir.source_scopes.len());
}

fn make_mir_scope<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
    mir: &Body<'tcx>,
    variables: &Option<DenseBitSet<SourceScope>>,
    debug_context: &mut FunctionDebugContext<'tcx, &'ll DIScope, &'ll DILocation>,
    instantiated: &mut DenseBitSet<SourceScope>,
    discriminators: &mut FxHashMap<BytePos, u32>,
    scope: SourceScope,
) {
    if instantiated.contains(scope) {
        return;
    }

    let scope_data = &mir.source_scopes[scope];
    let parent_scope = if let Some(parent) = scope_data.parent_scope {
        make_mir_scope(
            cx,
            instance,
            mir,
            variables,
            debug_context,
            instantiated,
            discriminators,
            parent,
        );
        debug_context.scopes[parent]
    } else {
        // The root is the function itself.
        let file = cx.sess().source_map().lookup_source_file(mir.span.lo());
        debug_context.scopes[scope] = DebugScope {
            file_start_pos: file.start_pos,
            file_end_pos: file.end_position(),
            ..debug_context.scopes[scope]
        };
        instantiated.insert(scope);
        return;
    };

    if let Some(vars) = variables
        && !vars.contains(scope)
        && scope_data.inlined.is_none()
    {
        // Do not create a DIScope if there are no variables defined in this
        // MIR `SourceScope`, and it's not `inlined`, to avoid debuginfo bloat.
        debug_context.scopes[scope] = parent_scope;
        instantiated.insert(scope);
        return;
    }

    let loc = cx.lookup_debug_loc(scope_data.span.lo());
    let file_metadata = file_metadata(cx, &loc.file);

    let dbg_scope = match scope_data.inlined {
        Some((callee, _)) => {
            // FIXME(eddyb) this would be `self.monomorphize(&callee)`
            // if this is moved to `rustc_codegen_ssa::mir::debuginfo`.
            let callee = cx.tcx.instantiate_and_normalize_erasing_regions(
                instance.args,
                cx.typing_env(),
                ty::EarlyBinder::bind(callee),
            );
            debug_context.inlined_function_scopes.entry(callee).or_insert_with(|| {
                let callee_fn_abi = cx.fn_abi_of_instance(callee, ty::List::empty());
                cx.dbg_scope_fn(callee, callee_fn_abi, None)
            })
        }
        None => unsafe {
            llvm::LLVMDIBuilderCreateLexicalBlock(
                DIB(cx),
                parent_scope.dbg_scope,
                file_metadata,
                loc.line,
                loc.col,
            )
        },
    };

    let inlined_at = scope_data.inlined.map(|(_, callsite_span)| {
        let callsite_span = hygiene::walk_chain_collapsed(callsite_span, mir.span);
        let callsite_scope = parent_scope.adjust_dbg_scope_for_span(cx, callsite_span);
        let loc = cx.dbg_loc(callsite_scope, parent_scope.inlined_at, callsite_span);

        // NB: In order to produce proper debug info for variables (particularly
        // arguments) in multiply-inlined functions, LLVM expects to see a single
        // DILocalVariable with multiple different DILocations in the IR. While
        // the source information for each DILocation would be identical, their
        // inlinedAt attributes will be unique to the particular callsite.
        //
        // We generate DILocations here based on the callsite's location in the
        // source code. A single location in the source code usually can't
        // produce multiple distinct calls so this mostly works, until
        // macros get involved. A macro can generate multiple calls
        // at the same span, which breaks the assumption that we're going to
        // produce a unique DILocation for every scope we process here. We
        // have to explicitly add discriminators if we see inlines into the
        // same source code location.
        //
        // Note further that we can't key this hashtable on the span itself,
        // because these spans could have distinct SyntaxContexts. We have
        // to key on exactly what we're giving to LLVM.
        match discriminators.entry(callsite_span.lo()) {
            Entry::Occupied(mut o) => {
                *o.get_mut() += 1;
                // NB: We have to emit *something* here or we'll fail LLVM IR verification
                // in at least some circumstances (see issue #135322) so if the required
                // discriminant cannot be encoded fall back to the dummy location.
                unsafe { llvm::LLVMRustDILocationCloneWithBaseDiscriminator(loc, *o.get()) }
                    .unwrap_or_else(|| {
                        cx.dbg_loc(callsite_scope, parent_scope.inlined_at, DUMMY_SP)
                    })
            }
            Entry::Vacant(v) => {
                v.insert(0);
                loc
            }
        }
    });

    debug_context.scopes[scope] = DebugScope {
        dbg_scope,
        inlined_at: inlined_at.or(parent_scope.inlined_at),
        file_start_pos: loc.file.start_pos,
        file_end_pos: loc.file.end_position(),
    };
    instantiated.insert(scope);
}
