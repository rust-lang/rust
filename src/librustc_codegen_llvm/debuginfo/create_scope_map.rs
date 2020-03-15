use super::metadata::{file_metadata, UNKNOWN_COLUMN_NUMBER, UNKNOWN_LINE_NUMBER};
use super::utils::DIB;
use rustc_codegen_ssa::mir::debuginfo::{DebugScope, FunctionDebugContext};

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DIScope, DISubprogram};
use rustc::mir::{Body, SourceScope};

use rustc_index::bit_set::BitSet;
use rustc_index::vec::Idx;

/// Produces DIScope DIEs for each MIR Scope which has variables defined in it.
pub fn compute_mir_scopes(
    cx: &CodegenCx<'ll, '_>,
    mir: &Body<'_>,
    fn_metadata: &'ll DISubprogram,
    debug_context: &mut FunctionDebugContext<&'ll DIScope>,
) {
    // Find all the scopes with variables defined in them.
    let mut has_variables = BitSet::new_empty(mir.source_scopes.len());
    // FIXME(eddyb) take into account that arguments always have debuginfo,
    // irrespective of their name (assuming full debuginfo is enabled).
    for var_debug_info in &mir.var_debug_info {
        has_variables.insert(var_debug_info.source_info.scope);
    }

    // Instantiate all scopes.
    for idx in 0..mir.source_scopes.len() {
        let scope = SourceScope::new(idx);
        make_mir_scope(cx, &mir, fn_metadata, &has_variables, debug_context, scope);
    }
}

fn make_mir_scope(
    cx: &CodegenCx<'ll, '_>,
    mir: &Body<'_>,
    fn_metadata: &'ll DISubprogram,
    has_variables: &BitSet<SourceScope>,
    debug_context: &mut FunctionDebugContext<&'ll DISubprogram>,
    scope: SourceScope,
) {
    if debug_context.scopes[scope].is_valid() {
        return;
    }

    let scope_data = &mir.source_scopes[scope];
    let parent_scope = if let Some(parent) = scope_data.parent_scope {
        make_mir_scope(cx, mir, fn_metadata, has_variables, debug_context, parent);
        debug_context.scopes[parent]
    } else {
        // The root is the function itself.
        let loc = cx.lookup_debug_loc(mir.span.lo());
        debug_context.scopes[scope] = DebugScope {
            scope_metadata: Some(fn_metadata),
            file_start_pos: loc.file.start_pos,
            file_end_pos: loc.file.end_pos,
        };
        return;
    };

    if !has_variables.contains(scope) {
        // Do not create a DIScope if there are no variables
        // defined in this MIR Scope, to avoid debuginfo bloat.
        debug_context.scopes[scope] = parent_scope;
        return;
    }

    let loc = cx.lookup_debug_loc(scope_data.span.lo());
    let file_metadata = file_metadata(cx, &loc.file.name, debug_context.defining_crate);

    let scope_metadata = unsafe {
        Some(llvm::LLVMRustDIBuilderCreateLexicalBlock(
            DIB(cx),
            parent_scope.scope_metadata.unwrap(),
            file_metadata,
            loc.line.unwrap_or(UNKNOWN_LINE_NUMBER),
            loc.col.unwrap_or(UNKNOWN_COLUMN_NUMBER),
        ))
    };
    debug_context.scopes[scope] = DebugScope {
        scope_metadata,
        file_start_pos: loc.file.start_pos,
        file_end_pos: loc.file.end_pos,
    };
}
