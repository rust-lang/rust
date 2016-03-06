// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use llvm::{ValueRef};
use llvm;
use middle::weak_lang_items;
use trans::base::{llvm_linkage_by_name};
use trans::common::*;
use trans::declare;
use trans::type_of;
use middle::ty;

use syntax::attr;
use syntax::parse::token::{InternedString};
use syntax::ast;
use syntax::attr::AttrMetaMethods;

use rustc_front::hir;

///////////////////////////////////////////////////////////////////////////
// Calls to external functions

pub fn register_static(ccx: &CrateContext,
                       foreign_item: &hir::ForeignItem) -> ValueRef {
    let ty = ccx.tcx().node_id_to_type(foreign_item.id);
    let llty = type_of::type_of(ccx, ty);

    let ident = link_name(foreign_item.name, &foreign_item.attrs);
    let c = match attr::first_attr_value_str_by_name(&foreign_item.attrs,
                                                     "linkage") {
        // If this is a static with a linkage specified, then we need to handle
        // it a little specially. The typesystem prevents things like &T and
        // extern "C" fn() from being non-null, so we can't just declare a
        // static and call it a day. Some linkages (like weak) will make it such
        // that the static actually has a null value.
        Some(name) => {
            let linkage = match llvm_linkage_by_name(&name) {
                Some(linkage) => linkage,
                None => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "invalid linkage specified");
                }
            };
            let llty2 = match ty.sty {
                ty::TyRawPtr(ref mt) => type_of::type_of(ccx, mt.ty),
                _ => {
                    ccx.sess().span_fatal(foreign_item.span,
                                          "must have type `*T` or `*mut T`");
                }
            };
            unsafe {
                // Declare a symbol `foo` with the desired linkage.
                let g1 = declare::declare_global(ccx, &ident[..], llty2);
                llvm::SetLinkage(g1, linkage);

                // Declare an internal global `extern_with_linkage_foo` which
                // is initialized with the address of `foo`.  If `foo` is
                // discarded during linking (for example, if `foo` has weak
                // linkage and there are no definitions), then
                // `extern_with_linkage_foo` will instead be initialized to
                // zero.
                let mut real_name = "_rust_extern_with_linkage_".to_string();
                real_name.push_str(&ident);
                let g2 = declare::define_global(ccx, &real_name[..], llty).unwrap_or_else(||{
                    ccx.sess().span_fatal(foreign_item.span,
                                          &format!("symbol `{}` is already defined", ident))
                });
                llvm::SetLinkage(g2, llvm::InternalLinkage);
                llvm::LLVMSetInitializer(g2, g1);
                g2
            }
        }
        None => // Generate an external declaration.
            declare::declare_global(ccx, &ident[..], llty),
    }
}

///////////////////////////////////////////////////////////////////////////
// General ABI Support
//
// This code is kind of a confused mess and needs to be reworked given
// the massive simplifications that have occurred.

pub fn link_name(name: ast::Name, attrs: &[ast::Attribute]) -> InternedString {
    match attr::first_attr_value_str_by_name(attrs, "link_name") {
        Some(ln) => ln.clone(),
        None => match weak_lang_items::link_name(attrs) {
            Some(name) => name,
            None => name.as_str(),
        }
    }
}
