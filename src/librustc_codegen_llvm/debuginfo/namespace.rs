// Namespace Handling.

use super::metadata::{unknown_file_metadata, UNKNOWN_LINE_NUMBER};

use crate::debuginfo::CrateDebugContext;
use crate::llvm;
use crate::llvm::debuginfo::DIScope;
use rustc::hir::map::DefPathData;
use rustc_hir::def_id::DefId;

use rustc_data_structures::small_c_str::SmallCStr;

pub fn item_namespace(
    dbg_cx: &CrateDebugContext<'ll, '_>,
    def_id: DefId,
) -> &'ll DIScope {
    if let Some(&scope) = dbg_cx.namespace_map.borrow().get(&def_id) {
        return scope;
    }

    let def_key = dbg_cx.tcx.def_key(def_id);
    let parent_scope = def_key.parent.map(|parent| {
        item_namespace(dbg_cx, DefId { krate: def_id.krate, index: parent })
    });

    let namespace_name = match def_key.disambiguated_data.data {
        DefPathData::CrateRoot => dbg_cx.tcx.crate_name(def_id.krate),
        data => data.as_symbol(),
    };

    let namespace_name = SmallCStr::new(&namespace_name.as_str());

    let scope = unsafe {
        llvm::LLVMRustDIBuilderCreateNameSpace(
            dbg_cx.builder,
            parent_scope,
            namespace_name.as_ptr(),
            unknown_file_metadata(dbg_cx),
            UNKNOWN_LINE_NUMBER,
        )
    };

    dbg_cx.namespace_map.borrow_mut().insert(def_id, scope);
    scope
}
