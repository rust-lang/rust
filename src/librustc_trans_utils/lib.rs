// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(i128_type)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_patterns)]
#![feature(conservative_impl_trait)]

extern crate ar;
extern crate flate2;
#[macro_use]
extern crate log;

#[macro_use]
extern crate rustc;
extern crate rustc_back;
extern crate syntax;
extern crate syntax_pos;
extern crate rustc_data_structures;

use rustc::ty::{TyCtxt, Instance};
use rustc::hir;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::hir::map as hir_map;
use rustc::util::nodemap::NodeSet;

pub mod link;
pub mod trans_crate;

/// check for the #[rustc_error] annotation, which forces an
/// error in trans. This is used to write compile-fail tests
/// that actually test that compilation succeeds without
/// reporting an error.
pub fn check_for_rustc_errors_attr(tcx: TyCtxt) {
    if let Some((id, span)) = *tcx.sess.entry_fn.borrow() {
        let main_def_id = tcx.hir.local_def_id(id);

        if tcx.has_attr(main_def_id, "rustc_error") {
            tcx.sess.span_fatal(span, "compilation successful");
        }
    }
}

/// The context provided lists a set of reachable ids as calculated by
/// middle::reachable, but this contains far more ids and symbols than we're
/// actually exposing from the object file. This function will filter the set in
/// the context to the set of ids which correspond to symbols that are exposed
/// from the object file being generated.
///
/// This list is later used by linkers to determine the set of symbols needed to
/// be exposed from a dynamic library and it's also encoded into the metadata.
pub fn find_exported_symbols<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> NodeSet {
    tcx.reachable_set(LOCAL_CRATE).0.iter().cloned().filter(|&id| {
        // Next, we want to ignore some FFI functions that are not exposed from
        // this crate. Reachable FFI functions can be lumped into two
        // categories:
        //
        // 1. Those that are included statically via a static library
        // 2. Those included otherwise (e.g. dynamically or via a framework)
        //
        // Although our LLVM module is not literally emitting code for the
        // statically included symbols, it's an export of our library which
        // needs to be passed on to the linker and encoded in the metadata.
        //
        // As a result, if this id is an FFI item (foreign item) then we only
        // let it through if it's included statically.
        match tcx.hir.get(id) {
            hir_map::NodeForeignItem(..) => {
                let def_id = tcx.hir.local_def_id(id);
                tcx.is_statically_included_foreign_item(def_id)
            }

            // Only consider nodes that actually have exported symbols.
            hir_map::NodeItem(&hir::Item {
                node: hir::ItemStatic(..), .. }) |
            hir_map::NodeItem(&hir::Item {
                node: hir::ItemFn(..), .. }) |
            hir_map::NodeImplItem(&hir::ImplItem {
                node: hir::ImplItemKind::Method(..), .. }) => {
                let def_id = tcx.hir.local_def_id(id);
                let generics = tcx.generics_of(def_id);
                (generics.parent_types == 0 && generics.types.is_empty()) &&
                // Functions marked with #[inline] are only ever translated
                // with "internal" linkage and are never exported.
                !Instance::mono(tcx, def_id).def.requires_local(tcx)
            }

            _ => false
        }
    }).collect()
}
