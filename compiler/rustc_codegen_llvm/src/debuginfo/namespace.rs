// Namespace Handling.

use super::utils::{debug_context, DIB};
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_middle::ty::{self, Instance};

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::DIScope;
use rustc_hir::def_id::DefId;

pub fn mangled_name_of_instance<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    instance: Instance<'tcx>,
) -> ty::SymbolName<'tcx> {
    let tcx = cx.tcx;
    tcx.symbol_name(instance)
}

pub fn item_namespace<'ll>(cx: &CodegenCx<'ll, '_>, def_id: DefId) -> &'ll DIScope {
    if let Some(&scope) = debug_context(cx).namespace_map.borrow().get(&def_id) {
        return scope;
    }

    let def_key = cx.tcx.def_key(def_id);
    let parent_scope = def_key
        .parent
        .map(|parent| item_namespace(cx, DefId { krate: def_id.krate, index: parent }));

    let namespace_name_string = {
        let mut output = String::new();
        type_names::push_item_name(cx.tcx, def_id, false, &mut output);
        output
    };

    let scope = unsafe {
        llvm::LLVMRustDIBuilderCreateNameSpace(
            DIB(cx),
            parent_scope,
            namespace_name_string.as_ptr().cast(),
            namespace_name_string.len(),
            false, // ExportSymbols (only relevant for C++ anonymous namespaces)
        )
    };

    debug_context(cx).namespace_map.borrow_mut().insert(def_id, scope);
    scope
}
