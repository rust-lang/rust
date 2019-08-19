// Namespace Handling.

use super::metadata::{unknown_file_metadata, UNKNOWN_LINE_NUMBER};
use super::utils::{DIB, debug_context};
use rustc::ty::{self, Instance};

use crate::llvm;
use crate::llvm::debuginfo::DIScope;
use crate::common::CodegenCx;
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;

use rustc_data_structures::small_c_str::SmallCStr;

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
    let parent_scope = def_key.parent.map(|parent| {
        item_namespace(cx, DefId {
            krate: def_id.krate,
            index: parent
        })
    });

    let namespace_name = match def_key.disambiguated_data.data {
        DefPathData::CrateRoot => cx.tcx.crate_name(def_id.krate).as_str(),
        data => data.as_interned_str().as_str()
    };

    let namespace_name = SmallCStr::new(&namespace_name);

    let scope = unsafe {
        llvm::LLVMRustDIBuilderCreateNameSpace(
            DIB(cx),
            parent_scope,
            namespace_name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER)
    };

    debug_context(cx).namespace_map.borrow_mut().insert(def_id, scope);
    scope
}
