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
use syntax::ast;

use llvm;
use llvm::debuginfo::DIScope;
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use common::CodegenCx;

use std::ffi::CString;
use std::ptr;

pub fn mangled_name_of_instance<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    instance: Instance<'tcx>,
) -> ty::SymbolName {
     let tcx = cx.tcx;
     tcx.symbol_name(instance)
}

pub fn mangled_name_of_item<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    node_id: ast::NodeId,
) -> ty::SymbolName {
    let tcx = cx.tcx;
    let node_def_id = tcx.hir.local_def_id(node_id);
    let instance = Instance::mono(tcx, node_def_id);
    tcx.symbol_name(instance)
}

pub fn item_namespace(cx: &CodegenCx, def_id: DefId) -> DIScope {
    if let Some(&scope) = debug_context(cx).namespace_map.borrow().get(&def_id) {
        return scope;
    }

    let def_key = cx.tcx.def_key(def_id);
    let parent_scope = def_key.parent.map_or(ptr::null_mut(), |parent| {
        item_namespace(cx, DefId {
            krate: def_id.krate,
            index: parent
        })
    });

    let namespace_name = match def_key.disambiguated_data.data {
        DefPathData::CrateRoot => cx.tcx.crate_name(def_id.krate).as_str(),
        data => data.as_interned_str()
    };

    let namespace_name = CString::new(namespace_name.as_bytes()).unwrap();

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
