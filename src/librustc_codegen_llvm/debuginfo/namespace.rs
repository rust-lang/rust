// Namespace Handling.

use super::utils::{debug_context, DIB};
use rustc_middle::ty::{self, Instance};

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::DIScope;
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::DefPathData;

pub fn mangled_name_of_instance<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    instance: Instance<'tcx>,
) -> ty::SymbolName {
    let tcx = cx.tcx;
    tcx.symbol_name(instance)
}

pub fn item_namespace(cx: &CodegenCx<'ll, '_>, def_id: DefId) -> &'ll DIScope {
    if let Some(&scope) = debug_context(cx).namespace_map.borrow().get(&def_id) {
        return scope;
    }

    let def_key = cx.tcx.def_key(def_id);
    let parent_scope = def_key
        .parent
        .map(|parent| item_namespace(cx, DefId { krate: def_id.krate, index: parent }));

    let namespace_name = match def_key.disambiguated_data.data {
        DefPathData::CrateRoot => cx.tcx.crate_name(def_id.krate),
        data => data.as_symbol(),
    };
    let namespace_name = namespace_name.as_str();

    let scope = unsafe {
        llvm::LLVMRustDIBuilderCreateNameSpace(
            DIB(cx),
            parent_scope,
            namespace_name.as_ptr().cast(),
            namespace_name.len(),
            false, // ExportSymbols (only relevant for C++ anonymous namespaces)
        )
    };

    debug_context(cx).namespace_map.borrow_mut().insert(def_id, scope);
    scope
}
