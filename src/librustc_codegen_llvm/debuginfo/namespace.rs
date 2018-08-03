// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Namespace Handling.

use super::metadata::{unknown_file_metadata, UNKNOWN_LINE_NUMBER};
use super::utils::{DIB, debug_context};
use monomorphize::Instance;
use rustc::ty;

use llvm;
use llvm::debuginfo::DIScope;
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use common::CodegenCx;
use value::Value;

use rustc_data_structures::small_c_str::SmallCStr;

pub fn mangled_name_of_instance<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx, &'a Value>,
    instance: Instance<'tcx>,
) -> ty::SymbolName {
     let tcx = cx.tcx;
     tcx.symbol_name(instance)
}

pub fn item_namespace(cx: &CodegenCx<'ll, '_, &'ll Value>, def_id: DefId) -> &'ll DIScope {
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
