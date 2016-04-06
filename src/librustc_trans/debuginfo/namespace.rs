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

use super::utils::{DIB, debug_context};

use llvm;
use llvm::debuginfo::DIScope;
use rustc::hir::def_id::DefId;
use rustc::hir::map as hir_map;
use common::CrateContext;

use std::ffi::CString;
use std::iter::once;
use std::ptr;
use std::rc::{Rc, Weak};
use syntax::ast;
use syntax::parse::token;

pub struct NamespaceTreeNode {
    pub name: ast::Name,
    pub scope: DIScope,
    pub parent: Option<Weak<NamespaceTreeNode>>,
}

impl NamespaceTreeNode {
    pub fn mangled_name_of_contained_item(&self, item_name: &str) -> String {
        fn fill_nested(node: &NamespaceTreeNode, output: &mut String) {
            match node.parent {
                Some(ref parent) => fill_nested(&parent.upgrade().unwrap(), output),
                None => {}
            }
            let string = node.name.as_str();
            output.push_str(&string.len().to_string());
            output.push_str(&string);
        }

        let mut name = String::from("_ZN");
        fill_nested(self, &mut name);
        name.push_str(&item_name.len().to_string());
        name.push_str(item_name);
        name.push('E');
        name
    }
}

pub fn namespace_for_item(cx: &CrateContext, def_id: DefId) -> Rc<NamespaceTreeNode> {
    // prepend crate name.
    // This shouldn't need a roundtrip through InternedString.
    let krate = token::intern(&cx.tcx().crate_name(def_id.krate));
    let krate = hir_map::DefPathData::TypeNs(krate);
    let path = cx.tcx().def_path(def_id).data;
    let mut path = once(krate).chain(path.into_iter().map(|e| e.data)).peekable();

    let mut current_key = Vec::new();
    let mut parent_node: Option<Rc<NamespaceTreeNode>> = None;

    // Create/Lookup namespace for each element of the path.
    loop {
        // Emulate a for loop so we can use peek below.
        let path_element = match path.next() {
            Some(e) => e,
            None => break
        };
        // Ignore the name of the item (the last path element).
        if path.peek().is_none() {
            break;
        }

        // This shouldn't need a roundtrip through InternedString.
        let namespace_name = path_element.as_interned_str();
        let name = token::intern(&namespace_name);
        current_key.push(name);

        let existing_node = debug_context(cx).namespace_map.borrow()
                                             .get(&current_key).cloned();
        let current_node = match existing_node {
            Some(existing_node) => existing_node,
            None => {
                // create and insert
                let parent_scope = match parent_node {
                    Some(ref node) => node.scope,
                    None => ptr::null_mut()
                };
                let namespace_name = CString::new(namespace_name.as_bytes()).unwrap();
                let scope = unsafe {
                    llvm::LLVMDIBuilderCreateNameSpace(
                        DIB(cx),
                        parent_scope,
                        namespace_name.as_ptr(),
                        // cannot reconstruct file ...
                        ptr::null_mut(),
                        // ... or line information, but that's not so important.
                        0)
                };

                let node = Rc::new(NamespaceTreeNode {
                    name: name,
                    scope: scope,
                    parent: parent_node.map(|parent| Rc::downgrade(&parent)),
                });

                debug_context(cx).namespace_map.borrow_mut()
                                 .insert(current_key.clone(), node.clone());

                node
            }
        };

        parent_node = Some(current_node);
    }

    match parent_node {
        Some(node) => node,
        None => {
            bug!("debuginfo::namespace_for_item: path too short for {:?}", def_id);
        }
    }
}
