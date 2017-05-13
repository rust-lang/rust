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

use super::metadata::{file_metadata, unknown_file_metadata, UNKNOWN_LINE_NUMBER};
use super::utils::{DIB, debug_context, span_start};

use llvm;
use llvm::debuginfo::DIScope;
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use common::CrateContext;

use libc::c_uint;
use std::ffi::CString;
use std::ptr;
use syntax_pos::DUMMY_SP;

pub fn mangled_name_of_item(ccx: &CrateContext, def_id: DefId, extra: &str) -> String {
    fn fill_nested(ccx: &CrateContext, def_id: DefId, extra: &str, output: &mut String) {
        let def_key = ccx.tcx().def_key(def_id);
        if let Some(parent) = def_key.parent {
            fill_nested(ccx, DefId {
                krate: def_id.krate,
                index: parent
            }, "", output);
        }

        let name = match def_key.disambiguated_data.data {
            DefPathData::CrateRoot => ccx.tcx().crate_name(def_id.krate).as_str(),
            data => data.as_interned_str()
        };

        output.push_str(&(name.len() + extra.len()).to_string());
        output.push_str(&name);
        output.push_str(extra);
    }

    let mut name = String::from("_ZN");
    fill_nested(ccx, def_id, extra, &mut name);
    name.push('E');
    name
}

pub fn item_namespace(ccx: &CrateContext, def_id: DefId) -> DIScope {
    if let Some(&scope) = debug_context(ccx).namespace_map.borrow().get(&def_id) {
        return scope;
    }

    let def_key = ccx.tcx().def_key(def_id);
    let parent_scope = def_key.parent.map_or(ptr::null_mut(), |parent| {
        item_namespace(ccx, DefId {
            krate: def_id.krate,
            index: parent
        })
    });

    let namespace_name = match def_key.disambiguated_data.data {
        DefPathData::CrateRoot => ccx.tcx().crate_name(def_id.krate).as_str(),
        data => data.as_interned_str()
    };

    let namespace_name = CString::new(namespace_name.as_bytes()).unwrap();
    let span = ccx.tcx().def_span(def_id);
    let (file, line) = if span != DUMMY_SP {
        let loc = span_start(ccx, span);
        (file_metadata(ccx, &loc.file.name, &loc.file.abs_path), loc.line as c_uint)
    } else {
        (unknown_file_metadata(ccx), UNKNOWN_LINE_NUMBER)
    };

    let scope = unsafe {
        llvm::LLVMRustDIBuilderCreateNameSpace(
            DIB(ccx),
            parent_scope,
            namespace_name.as_ptr(),
            file,
            line as c_uint)
    };

    debug_context(ccx).namespace_map.borrow_mut().insert(def_id, scope);
    scope
}
